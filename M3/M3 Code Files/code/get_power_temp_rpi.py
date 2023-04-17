import telnetlib as tel
import time
import argparse
import gpiozero


def get_telnet_power(telnet_connection, last_power):
    """
    Read power values using telnet.
    """
    # Get the latest data available from the telnet connection without blocking
    tel_dat = str(telnet_connection.read_very_eager())
    print('telnet reading:', tel_dat)
    # Find the latest power measurement in the data
    idx = tel_dat.rfind('\n')
    idx2 = tel_dat[:idx].rfind('\n')
    idx2 = idx2 if idx2 != -1 else 0
    ln = tel_dat[idx2:idx].strip().split(',')
    if len(ln) < 2:
        total_power = last_power
    else:
        total_power = float(ln[-2])
    return total_power


def get_temps():
    cpu_temp = gpiozero.CPUTemperature().temperature
    return cpu_temp


# writing the temperature, power and cpu usage readings in csv

parser = argparse.ArgumentParser(description='temperature and power measurement code')
# added one argument to specify the location of csv file to store the output
parser.add_argument('--csv_loc', type=str, default='VGG11_readings.csv',
                    help='csv file to store the temperature and power measurements')
args = parser.parse_args()

with open(args.csv_loc, 'w+') as csv_file:
    csv_file.write(f"Time stamp,Power,avg_temp\n")
    # measurement   
    telnet_connection = tel.Telnet("192.168.4.1")
    total_power = 0.0
    start_time = time.time()

    while True:
        last_time = time.time()

        # system power
        total_power = get_telnet_power(telnet_connection, total_power)
        print('telnet power : ', total_power)

        # cpu temperature
        cpu_temp = get_temps()
        print('cpu temp : ', cpu_temp)

        # writing the readings to the csv file 
        time_stamp = last_time
        csv_file.write(f"{time_stamp}, {total_power}, {cpu_temp}\n")
        elapsed = time.time() - last_time
        DELAY = 0.2
        time.sleep(max(0., DELAY - elapsed))
