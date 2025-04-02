import serial
import time

def main():
    ser = serial.Serial("/dev/cu.usbmodem141301", 9600)

    time.sleep(2)

    ser.write(b'\x32')

    response = ser.read(1)
    print(response)

    time.sleep(2)

    ser.write(b'\x00')

    ser.close()

    # if setPosition(ser, 50):
    #     print("SUCCESS")
    # else: 
    #     print("FAILED")

    # ser.close()

# Sends a position value between 0 - 100
# Returns true if the message succeeds, otherwise returns false
# def setPosition(ser, value):
#     # Convert data to byte array
#     data_to_send = bytes([value])

#     # Write data
#     ser.write(data_to_send)

#     # Get response
#     response = ser.read(1)

#     print(response)

#     # Check if response is a success
#     return len(response) > 0 and response[0] == 0x06

if __name__ == "__main__":
    main()
