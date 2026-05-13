FILES = part1/ part2/ part3/

.PHONY: all check clean

all: handin.tar

check:
	@chmod +x check_submission.sh
	@./check_submission.sh

handin.tar: check $(FILES)
	tar cvf handin.tar --exclude="*.DS_Store" --exclude="__pycache__" --exclude=".ipynb_checkpoints" $(FILES)
	@echo "handin.tar is ready."

clean:
	rm -f *~ handin.tar
