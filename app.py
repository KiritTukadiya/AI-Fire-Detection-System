from flask import Flask, render_template
import requests
import os
import sys

from pathlib import Path
from ultralytics import YOLO


# ============================================================
# PROJECT ROOT
# ============================================================

AI_DIR = Path(__file__).resolve().parent


# ============================================================
# MAKE SURE PROJECT ROOT IS IN PYTHON PATH
# ============================================================

if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))


# ============================================================
# IMPORT RANDOM FOREST FUNCTIONS
# ============================================================

try:

    from predict import (
        load_model,
        predict_fire_status
    )

    print("\nSuccessfully imported predict.py")

except Exception as e:

    print("\nERROR importing predict.py:")
    print(e)
    raise


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# YOLO CONFIGURATION
# ============================================================

YOLO_MODEL_PATH = (
    AI_DIR
    / "yolo"
    / "runs"
    / "fire_smoke"
    / "weights"
    / "best.pt"
)


YOLO_IMAGE_PATH = (
    AI_DIR
    / "yolo"
    / "test_image.jpg"
)


# ============================================================
# THINGSPEAK CONFIGURATION
# ============================================================

CHANNEL_ID = "3519763"


THINGSPEAK_READ_API_KEY = os.getenv(
    "THINGSPEAK_READ_API_KEY",
    ""
)


THINGSPEAK_WRITE_API_KEY = os.getenv(
    "THINGSPEAK_WRITE_API_KEY",
    ""
)


THINGSPEAK_READ_URL = (
    f"https://api.thingspeak.com/channels/"
    f"{CHANNEL_ID}/feeds/last.json"
)


THINGSPEAK_WRITE_URL = (
    "https://api.thingspeak.com/update"
)


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

print("\n")
print("=" * 65)
print("LOADING RANDOM FOREST MODEL")
print("=" * 65)

try:

    RF_MODEL = load_model()

    print(
        "Random Forest model loaded successfully."
    )

except Exception as e:

    RF_MODEL = None

    print(
        "ERROR loading Random Forest model:"
    )

    print(e)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("\n")
print("=" * 65)
print("LOADING YOLO MODEL")
print("=" * 65)

print("YOLO Model Path:")
print(YOLO_MODEL_PATH)

print(
    "Model exists:",
    YOLO_MODEL_PATH.exists()
)

print("\nYOLO Image Path:")
print(YOLO_IMAGE_PATH)

print(
    "Image exists:",
    YOLO_IMAGE_PATH.exists()
)


YOLO_MODEL = None


try:

    if not YOLO_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nYOLO model not found:\n"
            f"{YOLO_MODEL_PATH}"
        )


    YOLO_MODEL = YOLO(
        str(YOLO_MODEL_PATH)
    )


    print(
        "\nYOLO model loaded successfully."
    )


except Exception as e:

    print(
        "\nERROR loading YOLO model:"
    )

    print(e)


# ============================================================
# GET SENSOR DATA FROM THINGSPEAK
# ============================================================

def get_sensor_data():

    try:

        # ----------------------------------------------------
        # CHECK API KEY
        # ----------------------------------------------------

        if not THINGSPEAK_READ_API_KEY:

            print("\n")
            print("=" * 65)

            print(
                "ERROR: THINGSPEAK_READ_API_KEY "
                "is not configured."
            )

            print("=" * 65)

            print(
                '\nSet it in PowerShell using:'
            )

            print(
                '$env:THINGSPEAK_READ_API_KEY="YOUR_READ_API_KEY"'
            )

            return None


        # ----------------------------------------------------
        # REQUEST
        # ----------------------------------------------------

        response = requests.get(

            THINGSPEAK_READ_URL,

            params={
                "api_key":
                    THINGSPEAK_READ_API_KEY
            },

            timeout=10
        )


        print("\n")
        print("-" * 65)

        print(
            "ThingSpeak READ Status:",
            response.status_code
        )

        print(
            "ThingSpeak READ Response:",
            response.text
        )

        print("-" * 65)


        response.raise_for_status()


        data = response.json()


        # ----------------------------------------------------
        # SENSOR FIELDS
        # ----------------------------------------------------

        temperature = data.get("field1")
        humidity = data.get("field2")
        gas = data.get("field3")
        flame = data.get("field4")


        # ----------------------------------------------------
        # CHECK DATA
        # ----------------------------------------------------

        if temperature is None:
            raise ValueError(
                "ThingSpeak field1 "
                "(temperature) is empty."
            )

        if humidity is None:
            raise ValueError(
                "ThingSpeak field2 "
                "(humidity) is empty."
            )

        if gas is None:
            raise ValueError(
                "ThingSpeak field3 "
                "(gas) is empty."
            )

        if flame is None:
            raise ValueError(
                "ThingSpeak field4 "
                "(flame) is empty."
            )


        # ----------------------------------------------------
        # CONVERT
        # ----------------------------------------------------

        temperature = float(temperature)
        humidity = float(humidity)
        gas = float(gas)
        flame = float(flame)


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print("\nSENSOR DATA")
        print("-" * 65)

        print(
            f"Temperature : {temperature:.2f} °C"
        )

        print(
            f"Humidity    : {humidity:.2f} %"
        )

        print(
            f"Gas         : {gas:.2f}"
        )

        print(
            f"Flame       : {flame}"
        )

        print("-" * 65)


        return (
            temperature,
            humidity,
            gas,
            flame
        )


    except requests.exceptions.RequestException as e:

        print(
            "\nThingSpeak READ ERROR:"
        )

        print(e)

        return None


    except Exception as e:

        print(
            "\nERROR reading sensor data:"
        )

        print(e)

        return None


# ============================================================
# RUN YOLO
# ============================================================

def run_yolo():

    fire_detected = False
    smoke_detected = False

    fire_confidence = 0.0
    smoke_confidence = 0.0


    print("\n")
    print("=" * 65)
    print("RUNNING YOLO FIRE DETECTION")
    print("=" * 65)


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if YOLO_MODEL is None:

        print(
            "ERROR: YOLO model is not loaded."
        )

        return (
            fire_detected,
            smoke_detected,
            fire_confidence,
            smoke_confidence
        )


    # --------------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------------

    if not YOLO_IMAGE_PATH.exists():

        print(
            "ERROR: YOLO test image not found:"
        )

        print(YOLO_IMAGE_PATH)

        return (
            fire_detected,
            smoke_detected,
            fire_confidence,
            smoke_confidence
        )


    print("\nYOLO image:")
    print(YOLO_IMAGE_PATH)


    # --------------------------------------------------------
    # RUN YOLO
    # --------------------------------------------------------

    try:

        results = YOLO_MODEL.predict(

            source=str(YOLO_IMAGE_PATH),

            conf=0.25,

            save=True,

            project=str(
                AI_DIR / "yolo" / "runs"
            ),

            name="flask_prediction",

            exist_ok=True
        )


        # ----------------------------------------------------
        # PROCESS DETECTIONS
        # ----------------------------------------------------

        for result in results:

            if result.boxes is None:
                continue


            for box in result.boxes:

                class_id = int(
                    box.cls[0].item()
                )


                confidence = float(
                    box.conf[0].item()
                )


                class_name = str(
                    YOLO_MODEL.names[class_id]
                ).lower()


                print(
                    f"YOLO Detection: "
                    f"{class_name} "
                    f"{confidence * 100:.2f}%"
                )


                # --------------------------------------------
                # FIRE
                # --------------------------------------------

                if class_name == "fire":

                    fire_detected = True

                    fire_confidence = max(
                        fire_confidence,
                        confidence
                    )


                # --------------------------------------------
                # SMOKE
                # --------------------------------------------

                elif class_name == "smoke":

                    smoke_detected = True

                    smoke_confidence = max(
                        smoke_confidence,
                        confidence
                    )


        # ----------------------------------------------------
        # YOLO RESULT
        # ----------------------------------------------------

        print("\n")
        print("-" * 65)
        print("YOLO RESULT")
        print("-" * 65)

        print(
            "Fire Detected:",
            fire_detected
        )

        print(
            "Smoke Detected:",
            smoke_detected
        )

        print(
            "Fire Confidence:",
            f"{fire_confidence * 100:.2f}%"
        )

        print(
            "Smoke Confidence:",
            f"{smoke_confidence * 100:.2f}%"
        )

        print("-" * 65)


        return (
            fire_detected,
            smoke_detected,
            fire_confidence,
            smoke_confidence
        )


    except Exception as e:

        print(
            "\nYOLO ERROR:"
        )

        print(e)


        return (
            False,
            False,
            0.0,
            0.0
        )


# ============================================================
# DECISION FUSION
# ============================================================
#
# RF + YOLO are combined here.
#
# RF:
#     Normal
#     Warning
#     Critical
#
# YOLO:
#     No Detection
#     Smoke
#     Fire
#
# Fusion priority:
#
#     Fire      -> Critical
#     RF Critical -> Critical
#     RF Warning + Smoke -> Warning
#     RF Warning -> Warning
#     Smoke -> Warning
#     Otherwise -> Normal
#
# ============================================================

def decision_fusion(
    rf_status,
    fire_detected,
    smoke_detected,
    fire_confidence=0.0,
    smoke_confidence=0.0
):

    print("\n")
    print("=" * 65)
    print("DECISION FUSION")
    print("=" * 65)


    print(
        "RF Status:",
        rf_status
    )

    print(
        "YOLO Fire:",
        fire_detected,
        f"({fire_confidence * 100:.2f}%)"
    )

    print(
        "YOLO Smoke:",
        smoke_detected,
        f"({smoke_confidence * 100:.2f}%)"
    )


    # --------------------------------------------------------
    # RULE 1
    # FIRE DETECTION = CRITICAL
    # --------------------------------------------------------

    if fire_detected:

        final_status = "Critical"


    # --------------------------------------------------------
    # RULE 2
    # RF CRITICAL = CRITICAL
    # --------------------------------------------------------

    elif rf_status == "Critical":

        final_status = "Critical"


    # --------------------------------------------------------
    # RULE 3
    # RF WARNING + SMOKE = WARNING
    # --------------------------------------------------------

    elif (
        rf_status == "Warning"
        and smoke_detected
    ):

        final_status = "Warning"


    # --------------------------------------------------------
    # RULE 4
    # RF WARNING
    # --------------------------------------------------------

    elif rf_status == "Warning":

        final_status = "Warning"


    # --------------------------------------------------------
    # RULE 5
    # YOLO SMOKE
    # --------------------------------------------------------

    elif smoke_detected:

        final_status = "Warning"


    # --------------------------------------------------------
    # RULE 6
    # BOTH NORMAL
    # --------------------------------------------------------

    else:

        final_status = "Normal"


    print(
        "\nFUSED FINAL STATUS:",
        final_status
    )

    print("=" * 65)


    return final_status


# ============================================================
# WRITE AI RESULTS TO THINGSPEAK
# ============================================================

def write_ai_results(
    rf_status,
    fire_detected,
    smoke_detected,
    final_status
):

    if not THINGSPEAK_WRITE_API_KEY:

        print(
            "\nWARNING:"
            " THINGSPEAK_WRITE_API_KEY "
            "is not configured."
        )

        return


    # --------------------------------------------------------
    # STATUS VALUES
    # --------------------------------------------------------

    status_values = {

        "Normal": 0,

        "Warning": 1,

        "Critical": 2
    }


    rf_value = status_values.get(
        rf_status,
        -1
    )


    final_value = status_values.get(
        final_status,
        -1
    )


    # --------------------------------------------------------
    # YOLO VALUE
    #
    # 0 = No Detection
    # 1 = Smoke
    # 2 = Fire
    # --------------------------------------------------------

    if fire_detected:

        yolo_value = 2

    elif smoke_detected:

        yolo_value = 1

    else:

        yolo_value = 0


    print("\n")
    print("=" * 65)
    print("UPLOADING AI RESULTS TO THINGSPEAK")
    print("=" * 65)


    print(
        "RF:",
        rf_status,
        "(",
        rf_value,
        ")"
    )


    print(
        "YOLO:",
        yolo_value
    )


    print(
        "Final:",
        final_status,
        "(",
        final_value,
        ")"
    )


    # --------------------------------------------------------
    # REQUEST
    # --------------------------------------------------------

    params = {

        "api_key":
            THINGSPEAK_WRITE_API_KEY,

        "field5":
            rf_value,

        "field6":
            yolo_value,

        "field7":
            final_value
    }


    try:

        response = requests.get(

            THINGSPEAK_WRITE_URL,

            params=params,

            timeout=10
        )


        print(
            "\nThingSpeak WRITE Status:",
            response.status_code
        )


        print(
            "ThingSpeak WRITE Response:",
            response.text
        )


        response.raise_for_status()


        if response.text.strip() == "0":

            print(
                "\nThingSpeak rejected "
                "the update."
            )

            print(
                "Possible reason:"
                " update interval is less "
                "than 15 seconds."
            )


        else:

            print(
                "\nAI results uploaded successfully."
            )

            print(
                "ThingSpeak Entry ID:",
                response.text
            )


    except Exception as e:

        print(
            "\nThingSpeak WRITE ERROR:"
        )

        print(e)


# ============================================================
# COMPLETE AI PIPELINE
# ============================================================

def run_complete_ai_pipeline():

    print("\n\n")

    print("=" * 65)

    print(
        "          AI FIRE DETECTION PIPELINE"
    )

    print(
        "          RF + YOLO + DECISION FUSION"
    )

    print("=" * 65)


    # ========================================================
    # STEP 1
    # SENSOR DATA
    # ========================================================

    sensor_data = get_sensor_data()


    if sensor_data is None:

        return {

            "temperature": "--",

            "humidity": "--",

            "gas": "--",

            "flame": "UNKNOWN",

            "rf_status": "Unknown",

            "yolo_status": "Unknown",

            "final_status": "Unknown"

        }


    (
        temperature,
        humidity,
        gas,
        flame
    ) = sensor_data


    # ========================================================
    # STEP 2
    # RANDOM FOREST
    # ========================================================

    rf_status = "Unknown"
    rf_probabilities = {}


    print("\n")

    print("=" * 65)

    print(
        "RANDOM FOREST PREDICTION"
    )

    print("=" * 65)


    if RF_MODEL is None:

        print(
            "Random Forest model "
            "is not loaded."
        )


    else:

        try:

            (
                rf_status,
                rf_probabilities
            ) = predict_fire_status(

                RF_MODEL,

                temperature,
                humidity,
                gas,
                flame

            )


            print(
                "RF Status:",
                rf_status
            )


            print(
                "RF Probabilities:",
                rf_probabilities
            )


        except Exception as e:

            print(
                "\nRF ERROR:"
            )

            print(e)


    # ========================================================
    # STEP 3
    # YOLO
    # ========================================================

    (
        fire_detected,
        smoke_detected,
        fire_confidence,
        smoke_confidence

    ) = run_yolo()


    # ========================================================
    # STEP 4
    # YOLO STATUS
    # ========================================================

    if fire_detected:

        yolo_status = "Fire"

    elif smoke_detected:

        yolo_status = "Smoke"

    else:

        yolo_status = "No Detection"


    # ========================================================
    # STEP 5
    # DECISION FUSION
    # ========================================================

    final_status = decision_fusion(

        rf_status,

        fire_detected,

        smoke_detected,

        fire_confidence,

        smoke_confidence

    )


    # ========================================================
    # STEP 6
    # FINAL RESULT
    # ========================================================

    print("\n")

    print("=" * 65)

    print(
        "                 FINAL RESULT"
    )

    print("=" * 65)


    print(
        f"Temperature : "
        f"{temperature:.2f} °C"
    )


    print(
        f"Humidity    : "
        f"{humidity:.2f} %"
    )


    print(
        f"Gas         : "
        f"{gas:.2f}"
    )


    print(
        "Flame       :",

        (
            "DETECTED"
            if flame == 1
            else "NOT DETECTED"
        )
    )


    print(
        "\nRF          :",
        rf_status
    )


    print(
        "YOLO        :",
        yolo_status
    )


    print(
        "Fusion      :",
        final_status
    )


    print("=" * 65)


    # ========================================================
    # STEP 7
    # WRITE TO THINGSPEAK
    # ========================================================

    write_ai_results(

        rf_status,

        fire_detected,

        smoke_detected,

        final_status

    )


    # ========================================================
    # STEP 8
    # RETURN DASHBOARD DATA
    # ========================================================

    return {

        "temperature":
            f"{temperature:.2f}",

        "humidity":
            f"{humidity:.2f}",

        "gas":
            f"{gas:.2f}",

        "flame":
            (
                "DETECTED"
                if flame == 1
                else "NOT DETECTED"
            ),

        "rf_status":
            rf_status,

        "yolo_status":
            yolo_status,

        "final_status":
            final_status

    }


# ============================================================
# DASHBOARD ROUTE
# ============================================================

@app.route("/")
def dashboard():

    data = run_complete_ai_pipeline()


    return render_template(

        "dashboard.html",

        data=data

    )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    print("\n")

    print("=" * 65)

    print(
        "       AI FIRE DETECTION DASHBOARD"
    )

    print(
        "       RF + YOLO + DECISION FUSION"
    )

    print("=" * 65)


    # --------------------------------------------------------
    # PROJECT
    # --------------------------------------------------------

    print(
        "\nProject Directory:"
    )

    print(AI_DIR)


    # --------------------------------------------------------
    # THINGSPEAK
    # --------------------------------------------------------

    print(
        "\nThingSpeak Channel:",
        CHANNEL_ID
    )


    print(
        "Read API Key configured:",
        bool(THINGSPEAK_READ_API_KEY)
    )


    print(
        "Write API Key configured:",
        bool(THINGSPEAK_WRITE_API_KEY)
    )


    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    print(
        "\nYOLO Model:"
    )

    print(
        YOLO_MODEL_PATH
    )


    print(
        "Model exists:",
        YOLO_MODEL_PATH.exists()
    )


    print(
        "\nYOLO Test Image:"
    )

    print(
        YOLO_IMAGE_PATH
    )


    print(
        "Image exists:",
        YOLO_IMAGE_PATH.exists()
    )


    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    print(
        "\nOpen dashboard:"
    )

    print(
        "http://127.0.0.1:5000"
    )


    print("=" * 65)


    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )