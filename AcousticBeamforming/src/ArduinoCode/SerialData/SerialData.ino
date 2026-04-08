#include <ADC.h>
#include <Arduino.h>

ADC *adc = new ADC(); // ADC library object

// Pin Teams (Balanced based on hardware capability)
const int team1[] = {A0, A1, A2, A3, A4, A5};   // Assigned to ADC1, Right side of BF
const int team2[] = {A6, A7, A10, A11, A12, A13}; // Assigned to ADC2, Left side of BF

volatile bool dataReady = false;
volatile int16_t buffers[12]; 
uint32_t header = 0xABCD1234;
IntervalTimer samplingTimer;

unsigned long lastBlink = 0;
bool ledState = false;

void setup() {

  Serial.begin(2000000);
  while (!Serial);
  
  // Configure ADC1
  adc->adc0->setAveraging(1); 
  adc->adc0->setResolution(12);
  adc->adc0->setConversionSpeed(ADC_CONVERSION_SPEED::VERY_HIGH_SPEED);
  adc->adc0->setSamplingSpeed(ADC_SAMPLING_SPEED::VERY_HIGH_SPEED);

  // Configure ADC2
  adc->adc1->setAveraging(1);
  adc->adc1->setResolution(12);
  adc->adc1->setConversionSpeed(ADC_CONVERSION_SPEED::VERY_HIGH_SPEED);
  adc->adc1->setSamplingSpeed(ADC_SAMPLING_SPEED::VERY_HIGH_SPEED);

  // Start timer at 44.1 kHz (22.67 microseconds)
  samplingTimer.begin(captureAllMics, 22.67);

  pinMode(LED_BUILTIN, OUTPUT);
}

void captureAllMics() {
  // We read two mics at the exact same time (one on each ADC)
  // This happens 6 times in a row
  for (int i = 0; i < 6; i++) {
    // Start conversions simultaneously
    adc->adc0->startSingleRead(team1[i]);
    adc->adc1->startSingleRead(team2[i]);

    // Wait for both to finish (very fast at VERY_HIGH_SPEED)
    while(adc->adc0->isConverting() || adc->adc1->isConverting());

    buffers[i] = adc->adc0->readSingle();     // Mic 1-6
    buffers[i+6] = adc->adc1->readSingle();   // Mic 7-12
  }
  
  dataReady = true;
}

void sendToPython() {
  Serial.write((uint8_t*)&header, 4);
  Serial.write((uint8_t*)buffers, sizeof(buffers));
  Serial.send_now();
}

void loop() {
  if (Serial && Serial.dtr()) {

    if (millis() - lastBlink > 500) {
      ledState = !ledState;
      digitalWrite(LED_BUILTIN, ledState);
      lastBlink = millis();
    }

    if (dataReady) {
      if (Serial.availableForWrite() >= (int)(sizeof(buffers) + 4)) {
        sendToPython();
      }
      dataReady = false;
    }
  } else {
    // Python is disconnected! 
    digitalWrite(LED_BUILTIN, LOW); 
    dataReady = false; // Just discard data so the buffer doesn't back up
  }
}