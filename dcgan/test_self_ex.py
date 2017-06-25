class test_self_ex(object):
    def __init__(self, a=1):
        # self._build_b()
        self.a = a

    def _build_b(self, b=2):
        self.b = b


def main():
    t = test_self_ex(a=3)
    print(t.b)


if '__main__' == __name__:
    main()