import matplotlib.pyplot as plt

f=open("..\\data\\cv\\model_log.txt")

def tran(line):
    i=0
    while line:
        if line[i]=='[':
            break
        else:
            i=i+1
    str=line[i+1:len(line)-2]
    y_=str.split(',')
    i=len(y_)
    y_=list(map(eval,y_))
    return y_

def main():
    lines=f.readlines()
    print(lines[0])
    print(lines[2])
    print(lines[3])
    
    fig=plt.figure()
    a1=plt.subplot(2,2,1)
    a2=plt.subplot(2,2,2)
    a3=plt.subplot(2,1,2)

    #loss曲线
    a1.plot(range(50),tran(lines[0]))
    a1.set_title('loss')
    a1.set_xlabel('epoch')
    a1.set_ylabel('loss')

    #训练accuracy曲线图
    a2.plot(range(50),tran(lines[4]))
    a2.set_title('train_accuracy')
    a2.set_xlabel('epoch')
    a2.set_ylabel('acc')

    #测试accuracy曲线图
    a3.plot(range(12),tran(lines[6]))
    a3.set_title('test_accuracy')
    a3.set_xlabel('epoch')
    a3.set_ylabel('acc')

    plt.show()

main()
