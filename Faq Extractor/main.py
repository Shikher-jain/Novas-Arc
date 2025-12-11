import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_command(cmd):
    logging.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        logging.error(f"Error while running: {cmd}")
        exit(1)

if __name__ == "__main__":
    run_command("python app.py")

    if "y" == input("Do you want to run the Streamlit app to ask questions? (y/n): ").strip().lower():
        run_command("streamlit run model.py")
    
