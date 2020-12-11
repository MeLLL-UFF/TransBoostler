
import sys
import os

import logging
logging.basicConfig(level=logging.INFO)


def main():
      
    runRDNB = bool(sys.argv[1])          # Curves using RDN-B
    runTransBoostler = bool(sys.argv[2]) # Curves using RDN-B using tranfer learning

    if(runTransBoostler):
        os.system('python transboostler_curves.py')

    if(runRDNB):
        os.system('python rdn_b_curves.py')

if __name__ == "__main__":
    sys.exit(main())