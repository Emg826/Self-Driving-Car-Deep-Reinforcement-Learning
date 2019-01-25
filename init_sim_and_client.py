import time
import subprocess
import signal

abs_file_path_to_city_env_bat_file = 'C:\\EricGabriel\\Unreal-AirSim-Environments\\CityEnviron\\CityEnviron\\Binaries\\Win64\\CityEnviron.exe  -ResX=640 -ResY=480 -windowed'
abs_file_path_to_python_client_file = 'C:\\EricGabriel\\Reinforcement-Learning-Project\\kerasrl_dqn_collision_avoidance.py'

# outer loop handles restarts while the
# inner loop handles check ups on the sim
secs_to_wait_for_sim_to_load_up = 30
secs_between_checks = 4
init_counter = 0
running_sim = None
running_client = None


while True:
  init_counter += 1
  # open the simulation
  print('Opening the airsim city environemnt')
  running_sim = subprocess.Popen(abs_file_path_to_city_env_bat_file, shell=True)

  print('Waiting {} seconds for the environment to get set up'.format(secs_to_wait_for_sim_to_load_up))
  time.sleep(secs_to_wait_for_sim_to_load_up)

  print('Starting the client')
  running_client = subprocess.Popen(abs_file_path_to_python_client_file, shell=True)

  want_to_break = False
  while True:
    time.sleep(secs_between_checks)

    if running_sim.poll() != None or running_client.poll() != None:
      running_sim.terminate()
      time.sleep(15)
      
      running_client.terminate()
      time.sleep(4)
      
      break
    
  print('just finished sim-client initialization {}'.format(init_counter))

