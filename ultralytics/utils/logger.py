class Logger:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def namestr(self, **kwargs):

        if self.verbose:
            for k, v in kwargs.items():
                print("%s = %s" % (k, repr(v)))
            print()

    def print(self, *args):
        if self.verbose:
            for msg in args:
                print(msg, end=" ")
            print()
