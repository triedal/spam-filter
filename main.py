from lib.model import Model
from lib.shell import Shell

if __name__ == '__main__':
    model = Model()
    model.train()
    model.evaluate()

    shell = Shell(model)
    shell.cmdloop('\nStarting interactive prompt...\n')
