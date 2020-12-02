import sys
import argparse
import h5py
import numpy as np
import os
import math as m


def ReadData(filename):
  gennames = []
  genval = []
  
  ii = 0
  file = open(filename,'r')
  while 1:
    ii += 1
    line = file.readline()
    if(ii<4):
      if(ii==2):
        gennames = line.strip().split()
        dtype = {'names' : gennames, 'formats' : ['i4']+(len(gennames)-1)*['f8']}
      elif(ii==3):
        genval = np.array(line.strip().split(),dtype=float)
    else:
      break
  file.close()
  
  head = np.rec.fromarrays(genval, dtype = dtype)
  data = np.genfromtxt(filename, skip_header=5, names=True)
  
  return head,data


def ReadGyre(filename):
  data = []
  
  ii = 0
  file = open(filename,'r')
  while 1:
    ii += 1
    line = file.readline()
    if not line:
      break
    elif(line.isspace()):
      continue
    elif(ii==1):
      head = list(np.array(line.strip().split(),dtype=float))
      head[0] = int(head[0])
      head[-1] = int(head[-1])
    else:
      line1 = line.replace('D','E')
      data1 = list(np.array(line1.strip().split(),dtype=float))
      data1[0] = int(data1[0])
      data.append(data1)
  head = np.array(head)
  data = np.array(data)
  file.close()
  
  return head,data


def prof2hdf5(filename):
  head,data = ReadData(filename)
  #hdfname = os.path.splitext(filename)[0]
  f = h5py.File(filename+".h5", "w")
  
  headnames = head.dtype.names
  for name in headnames:
    dset = f.create_dataset("head/"+name, data=head[name])
  datanames = data.dtype.names
  for name in datanames:
    dset = f.create_dataset("data/"+name, data=data[name])
  
  f.close()


def gyre2hdf5(filename):
  head,data = ReadGyre(filename)
  #hdfname = os.path.splitext(filename)[0]
  f = h5py.File(filename+".h5", "w")
  
  dset = f.create_dataset("head", data=head)
  dset = f.create_dataset("data", data=data)
  
  f.close()


def ReadProfHdf5(filename):
    f = h5py.File(filename, "r")
    
    head = dict(zip(f.attrs.keys(),f.attrs.values()))
    data = dict(zip(f.attrs.keys(),f.attrs.values()))
    #print data
    for ll in f['head']:
      head[ll] = f['head'][ll][...]

    complex_dtype = np.dtype([('re', '<f8'), ('im', '<f8')])
    
    # Convert the data into a record array and the global information in a dict
    
    local = []
    local_names = []
    
    for k in f['data']:
        # Convert items to complex
        
        if(f['data'][k].dtype == complex_dtype) :
            f['data'][k] = f['data'][k]['re'] + 1j*f['data'][k]['im']
        
        # Save array information to local record array, keep scalar information
        
        if not np.isscalar(f['data'][k]):
            local.append(f['data'][k][...])
            local_names.append(k)
            
    local = np.rec.fromarrays(local,names=local_names)
    
    # Return the global and local information
    
    f.close()
    
    return head, local

def ReadGyreHdf5(filename):
    f = h5py.File(filename, "r")
    
    head = f['head'][...]
    local = f['data'][...]
    
    f.close()
    
    return head, local
  

def hdf52prof(filename):
    """
    Convert a list of GYRE files to an ASCII file
    
    """
    gyre_files = [filename]
    ascii_file=hdfname = os.path.splitext(filename)[0] + '.data'
    ff = open(ascii_file, 'w')
    
    # Iterate over all GYRE files, and add the contents to the ASCII file
    for filenr, gyre_file in enumerate(gyre_files):
        #print("Writing file {} to {}".format(gyre_file, ascii_file))
        header,local = ReadProfHdf5(gyre_file)
        
        # If there is global information (header), print it out first
        out_header = []
        #if header:
            #for key in header:
                #out_header.append("{:10s} = {}".format(key,header[key]))
        
        if ascii_file is not None:
            for ii in np.arange(len(header)):
                string = str(int(ii+1))
                ff.write(string.rjust(40))
            ff.write('\n')
            for header_line in header:
                #ff.write('# {}\n'.format(header_line))
                ff.write(header_line.rjust(40))
            ff.write('\n')
            for key in header:
                #ff.write('# {}\n'.format(header_line))
                ff.write(str(header[key]).rjust(40))
            ff.write('\n\n')
        
        # then the local information:
        # the column names
        columns = ''
        
        # the data: we need to treat integers, floats and complex
        # numbers differently
        strfmt = ''
        counter = ''
        for i, col in enumerate(local.dtype.names):
            
            # For floats
            if isinstance(local[col][0],float):
                str1 = '{{{:d}:.16e}} '.format(i)
                str2 = '{} '.format(col)
                strfmt += str1.rjust(40)
                columns += str2.rjust(40)
                str3 = str(int(i+1))
                counter += str3.rjust(40)
                
                
            # For complex numbers
            elif isinstance(local[col][0],complex):
                str1 = '{{{:d}.real:.16e}} {{{:d}.imag:.16e}} '.format(i, i)
                str2 = 'Re({}) Im({}) '.format(col,col)
                strfmt += str1.rjust(40)
                columns += str2.rjust(40)
                str3 = str(int(i+1))
                counter += str3.rjust(40)
            
            # For integers
            elif isinstance(local[col][0],np.int32):
                str1 = '{{{:d}:d}} '.format(i)
                str2 = '{} '.format(col)
                strfmt += str1.rjust(40)
                columns += str2.rjust(40)
                str3 = str(int(i+1))
                counter += str3.rjust(40)
            
            else:
                raise ValueError
        counter += '\n'
        strfmt += '\n'            
        columns += '\n'
        
        # Then write all the data
        if ascii_file is not None and filenr == 0:
            ff.write(counter)
            ff.write(columns)
        
        for iline in range(len(local)):
            this_line = strfmt.format(*[local[col][iline] for col in \
                                                           local.dtype.names])
            
            if ascii_file is not None:
                ff.write(this_line)
            else:
                print(this_line.strip())
    
    # And close the file
    if ascii_file is not None:
        #print("Wrote contents to file {}".format(ascii_file))
        ff.close()



def hdf52gyre(filename):
    """
    Convert a list of GYRE files to an ASCII file
    
    """
    gyre_files = [filename]
    ascii_file = filename.strip().split('.')[0] + '.data.GYRE'
    ff = open(ascii_file, 'w')
    # Iterate over all GYRE files, and add the contents to the ASCII file
    for filenr, gyre_file in enumerate(gyre_files):
        #print("Writing file {} to {}".format(gyre_file, ascii_file))
        header,local = ReadGyreHdf5(gyre_file)
        
        if ascii_file is not None:
            for iih,head in enumerate(header):
              if(iih == 0):
                ff.write('  '+str(int(head)))
              elif(iih+1 != len(header)):
                exp = m.ceil(m.log10(head))
                head1 = head*10.**(-exp)
                str1 = '  %.12fE+' %head1
                ff.write(str1 + str(int(exp)))
              else:
                ff.write('    '+str(int(head)))
            ff.write('\n')
        
        
        ### UNCOMMENT THIS PART AND COMMENT THE NEXT WHEN RUNNING INTO PROBLEMS
        #for i in np.arange(len(local[:,0])):
          #line = ''
          #for j in np.arange(len(local[0,:])):
            #if(j == 0):
              #str1 = str(int(local[i,j]))
              #line += str1.rjust(6)
            #else:
              #str1 = str('%.16e' %local[i,j])
              #str2 = str1.replace("e","D")
              #line += str2.rjust(27)
          #line += '\n'
          #if ascii_file is not None:
            #ff.write(line)
        
        ######################################
        for i in np.arange(len(local[0,:])):
          line = ''
          for j in np.arange(len(local[:,0])):
            if(j == 0):
              str1 = str(int(local[j,i]))
              line += str1.rjust(6)
            else:
              str1 = str('%.16e' %local[j,i])
              str2 = str1.replace("e","D")
              line += str2.rjust(27)
          line += '\n'
          if ascii_file is not None:
            ff.write(line)
        #####################################
    
    # And close the file
    if ascii_file is not None:
        #print("Wrote contents to file {}".format(ascii_file))
        ff.close()
