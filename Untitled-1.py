for i in range(10):
    print(i)
    if open('stop.txt').read() == '1':
        break

print(open('stop.txt').read())