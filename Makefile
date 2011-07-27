all: executables

executables: trainExec classifyExec

trainExec: common
	make -C src/training

classifyExec: common
	make -C src/classification

common:
	make -C src/common

clean:
	@rm -rf obj

veryclean:
	@make clean
	@rm -rf bin
	@rm -rf lib