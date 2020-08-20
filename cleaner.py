'''
Vaccum Cleaner 
Akond Rahman 
Aug 20, 2020 
'''

import os

def doCleanUp(dir_name):
    pp_, non_pp = [], []
    for root_, dirs, files_ in os.walk(dir_name):
       for file_ in files_:
           full_p_file = os.path.join(root_, file_)
           if(os.path.exists(full_p_file)):
             if (full_p_file.endswith('.py')):
               pp_.append(full_p_file)
             else:
               non_pp.append(full_p_file)
    py_file_count = 0 
    dump_str = '' 
    for py_file in pp_:
        file_content = ''
        with open(py_file, 'r') as fil:
            file_content = fil.read()
        py_file_count += 1 
        dump_str = dump_str + '='*100
        dump_str = dump_str + 'FILENAME:::' + py_file + ':::' 
        dump_str = dump_str + '*'*25
        dump_str = dump_str + 'COUNT:::' + str(py_file_count)
        dump_str = dump_str + '*'*25
        dump_str = dump_str + file_content        
        dump_str = dump_str + '*'*25
        dump_str = dump_str + '='*100        

        
    dumpContentIntoFile( dump_str, 'QUAL.DATASET.RQ1.txt'  )
    # for f_ in non_pp:
    #     os.remove(f_)
    print("="*50)
    print(dir_name)
    print('removed {} non-python files, kept {} Python files #savespace '.format(len(non_pp), len(pp_)) )
    print("="*50 )


def dumpContentIntoFile(strP, fileP):
    fileToWrite = open( fileP, 'w')
    fileToWrite.write(strP )
    fileToWrite.close()
    return str(os.stat(fileP).st_size)

if __name__ == '__main__':
    dir_ = '/Users/arahman/FSE2021_ML_REPOS/MODELZOO/'
    doCleanUp(dir_)  