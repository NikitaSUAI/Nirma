class BaseTask:

    def __init__(self):
        pass

    def process(self, data):
        return data

    def check_input(self, data):
        return True

    def check_output(self, data):
        return True

    @staticmethod
    def init_from_config(config):
        pass


