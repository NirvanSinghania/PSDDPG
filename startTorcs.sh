#!/bin/bash
while true;
do
	ps cax | grep torcs > /dev/null
	if [ $? -eq 0 ]; then
	  : #echo "Process is running."
	else
	  echo "Process is not running."
      #Replace vtorcs with your TORCS directory and sh autostart.sh give absolute path
	  cd  ~/vtorcs  && ./torcs   & sleep 2 & sh  ~/autostart.sh
          	  	  
	fi
done;


