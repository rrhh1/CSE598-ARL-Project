#include <Arduino.h>

const int DIR_PIN = 2;
const int PUL_PIN = 3;
const int EN_PIN = 4;

const int BUTTON1_PIN = 5;
const int BUTTON2_PIN = 6;

const int TOTAL_STEPS = 17600;

unsigned char incomingByte = 0;

int currentPosition = 0;

void setup() {
    // Outputs
    pinMode(PUL_PIN, OUTPUT);
    pinMode(DIR_PIN, OUTPUT); 
    pinMode(EN_PIN, OUTPUT);

    // Inputs
    pinMode(BUTTON1_PIN, INPUT);
    pinMode(BUTTON2_PIN, INPUT);

    // Set Enable to low (active)
    digitalWrite(EN_PIN, LOW);
    // Set Direction to left
    digitalWrite(DIR_PIN, HIGH);

    // LED for debbugging
    pinMode(LED_BUILTIN, OUTPUT);

    // Setup serial
    Serial.begin(9600);
}

void loop() {
    if (Serial.available() > 0) {
        // read the incoming byte
        incomingByte = Serial.read();

        digitalWrite(LED_BUILTIN, HIGH);
        
        // send ack
        Serial.write(0x06);

        // calculate step change
        int newPosition = incomingByte / 100.0 * 18500;
        int stepsChange = newPosition - currentPosition;
        currentPosition = newPosition;

        // set direction
        if(stepsChange < 0) {
            digitalWrite(DIR_PIN, LOW); 
        } else {
            digitalWrite(DIR_PIN, HIGH);
        }

        // send signal to motor
        for(int i = 0; i < abs(stepsChange); i++) {
            digitalWrite(PUL_PIN, LOW);
            delayMicroseconds(50);
            digitalWrite(PUL_PIN, HIGH);
            delayMicroseconds(50); 
        }
    }
}