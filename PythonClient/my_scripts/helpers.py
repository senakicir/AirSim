from PythonClient import *

def SaveBonePositions2(index, bones, f_output):
    bones = [ v for v in bones.values() ]
    line = str(index)
    print(len(bones), 'lol')
    for i in range(0, len(bones)):
        line = line+'\t'+str(bones[i][b'x_val'])+'\t'+str(bones[i][b'y_val'])+'\t'+str(bones[i][b'z_val'])
    line = line+'\n'
    f_output.write(line)

def doNothing(x):
    pass
