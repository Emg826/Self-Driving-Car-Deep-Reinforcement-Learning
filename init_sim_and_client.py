
import time
import subprocess
import signal


abs_file_path_to_city_env_bat_file = 'C:\\EricGabriel\\Unreal-AirSim-Environments\\CityEnviron\\CityEnviron\\Binaries\\Win64\\CityEnviron.exe  -ResX=640 -ResY=480 -windowed'
abs_file_path_to_python_client_file = 'C:\\EricGabriel\\Reinforcement-Learning-Project\\kerasrl_dqn_collision_avoidance.py'



# outer loop handles restarts while the
# inner loop handles check ups on the sim
secs_to_wait_for_sim_to_load_up = 28
secs_between_checks = 10
init_counter = 0
while True:
  init_counter += 1
  # open the simulation
  print('Opening the airsim city environemnt')
  sim = subprocess.Popen(abs_file_path_to_city_env_bat_file, shell=True)

  print('Waiting {} seconds for the environment to get set up'.format(secs_to_wait_for_sim_to_load_up))
  time.sleep(secs_to_wait_for_sim_to_load_up)

  print('Starting the client')
  client = subprocess.Popen(abs_file_path_to_python_client_file, shell=True)

  while True:
    time.sleep(secs_between_checks)
    
    # run until see that sim has quit
    if sim.poll() != None:
      try:
        client.kill()
      except:
        client.terminate()
      break

    if client.poll() != None:
      try:
        sim.kill()
      except:
        sim.terminate()
      break
  print('just finished initialization {}'.format(init_counter))

