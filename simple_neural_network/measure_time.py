if __name__ == '__main__':
    import timeit
    print(timeit.timeit('neural_net(10000,0.7)', setup='from main import neural_net',number=100))
