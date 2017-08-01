rm -f assignment2.zip
zip -r assignment2.zip . -x "*.git*" "cifar10/*" "*cs231n/datasets*" "*.ipynb_checkpoints*" "*README.md" "*collectSubmission.sh" "*requirements.txt" ".env/*" "*.pyc" "*cs231n/build/*"
