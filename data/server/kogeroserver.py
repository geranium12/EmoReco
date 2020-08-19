"""
    KOGERO TEAM
"""

import sys
import multiprocessing as mulpro
import time
import socket
import pickle

MAX_CONNECTIONS = 10
PACKET_SIZE = 4096
ACCEPT_TIMEOUT = 0.5
CURRENT_VERSION = '0.3'
RECV_TIMEOUT = 30
SEND_TIMEOUT = 30
ERROR_TIMEOUT = 5
STR_ENCODING = 'UTF-8'


#----------------------------------
# Functions for manipulation with [ADMIN ~ SERVER] command pipe
#----------------------------------
    

def closeServer(input_pipe):
    
    input_pipe.send('#CS')
    print('Server Closed.')

def checkPipe(read_pipe):
    
    if (read_pipe.poll()):
        message = read_pipe.recv()
        if (message == '#CS'):
            return True
    return False


#----------------------------------
# Old function for getting message
#----------------------------------
    

"""
def getMessage(connection, adress):
       
    data = b''
    
    while (True):
        
        try:
            fragment = connection.recv(PACKET_SIZE)
        except:
            return ERROR_MSG
        
        data += fragment
        if (len(fragment) < PACKET_SIZE):
            break
    
    if (data == b''):
        data = ERROR_MSG
        
    return data
"""


#----------------------------------
# Functions for closing connections
#----------------------------------


def closeConnectionError(connection, adress):
    
    print('Error. ERROR connection closing with {}.'.format(adress))
    connection.settimeout(ERROR_TIMEOUT)
    try:
        connection.sendall(bytes('ERROR', STR_ENCODING))
    except:
        pass
    connection.close()
    sys.exit(0)

def closeConnectionSuccess(connection, adress):
    
    print('Successfull talk with {}. Closing Connection.'.format(adress))
    connection.settimeout(SEND_TIMEOUT)
    try:
        connection.sendall(bytes('END', STR_ENCODING))
    except:
        closeConnectionError(connection, adress)
    connection.close()
    sys.exit(0)
    
#----------------------------------
# Functions for sending data
#----------------------------------


def sendString(connection, adress, string, timeout = SEND_TIMEOUT):
    
    encoded_string = bytes(string, STR_ENCODING)
    sendRaw(connection, adress, encoded_string, timeout)
    return 1

def sendRaw(connection, adress, bytes_stream, timeout = SEND_TIMEOUT):
    
    connection.settimeout(timeout)
    try:
        connection.sendall(bytes_stream)
    except:
        closeConnectionError(connection, adress)
    return 1

#----------------------------------
# Functions for receiving data
#----------------------------------


def getString(connection, adress, timeout = RECV_TIMEOUT):
    
    encoded_string = getRaw(connection, adress, timeout)
    return str(encoded_string, STR_ENCODING)

def getRaw(connection, adress, timeout = RECV_TIMEOUT):
      
    data = b''
    connection.settimeout(timeout)
    while(True):
        try:
            fragment = connection.recv(PACKET_SIZE)
        except:
            closeConnectionError(connection, adress)
        data += fragment
        if (len(fragment) < PACKET_SIZE):
            break
    if (data == b''):
        closeConnectionError(connection, adress)
    return data


#----------------------------------
# Functions for handling types of connections
#----------------------------------
   
    
#New
def receiveStatistic(connection, adress):
    
    sendString(connection, adress, 'READY')
    pickled_stats = getRaw(connection, adress)
    with open('{} {} {}.pickle'.format(time.strftime('%Y.%m.%d %H.%M.%S',
                                              time.gmtime(time.time())),
                                adress[0], adress[1]), 'wb') as openfile:
        openfile.write(pickled_stats)
    
    received_dict = pickle.loads(pickled_stats)
    print('Received stats from {}.'.format(adress))
    print(str(received_dict['form']).replace('], ', '],\n'))
    print(str(received_dict['emotions']).replace('], ', '],\n'))
    print('-------------------------------------------')
    closeConnectionSuccess(connection, adress)
    return 1
        
"""
# Old
def receiveStatistic_OLD(connection, adress):
    
    connection.sendall(bytes('READY', 'UTF-8'))
    
    data = getMessage(connection, adress)
    
    if (data == ERROR_MSG):
        closeConnectionError(connection, adress)
        return ;
    
    received_object = pickle.loads(data)
    
    with open('{} {} {}.pickle'.format(time.strftime('%Y_%m_%d %H_%M_%S', time.gmtime(time.time())),
                                  adress[0], adress[1]), 'wb') as openfile:
        pickle.dump(received_object, openfile)
    
    connection.sendall(bytes('END', 'UTF-8'))
    connection.close()
    print('Received STATS from {}'.format(adress))
"""
   
def updateClient(connection, adress):

    data = b''
    sendString(connection, adress, CURRENT_VERSION)
    answer = getString(connection, adress)
    if (answer == 'NEXT'):
        try:
            with open('UPDATE {}.pickle'.format(str(CURRENT_VERSION)), 'rb') as openfile:
                data = openfile.read()
        except FileNotFoundError:
            print('UPDATE {}.pickle do not insist.'.format(CURRENT_VERSION))
            closeConnectionError(connection, adress)
        sendString(connection, adress, 'READY')
        answer = getString(connection, adress)
        if (answer != 'READY'):
            closeConnectionError(connection, adress)
        sendRaw(connection, adress, data)
        answer = getString(connection, adress)
        if (answer == 'END'):
            print('Successfull update for {}'.format(adress))
        else:
            print("Undefined response from {}, but update package has been sent.".format(adress))
    elif (answer == 'END'):
        print('{} has newer version of Network or client is Up-to-date. Aborting UPDATE.'.format(adress))
    else:
        closeConnectionError(connection, adress)
    closeConnectionSuccess(connection, adress)
    return 1
    
"""
#Old
def updateClient_OLD(connection, adress):
    
    connection.sendall(bytes(CURRENT_VERSION, 'UTF-8'))
    
    message = getMessage(connection, adress)
    if (message == ERROR_MSG):
        closeConnectionError(connection, adress)
        return ;
    
    elif (str(message, 'UTF-8') == 'NEXT'):
        
        with open('UPDATE {}.pickle'.format(CURRENT_VERSION), 'rb') as openfile:
            data = pickle.load(openfile)
        
        pickle_data = pickle.dumps(data)
        
        try:
            connection.sendall(pickle_data)
        except:
            closeConnectionError(connection, adress)
            return ;
        
        message = getMessage(connection, adress)
        if (str(message, 'UTF-8') == 'END'):
            print('Newer version of network sent to {}'.format(adress))
        else:
            print('Undefined error with {}, but looks like newer version of network was sent'.format(adress))
        
    else:
        print('{} have higher version'.format(adress))
    
    connection.close()
"""             


#----------------------------------
# Function for transfering each type of REQUEST for specialized function
#----------------------------------

  
def serveClient(connection, adress):
    
    connection.settimeout(RECV_TIMEOUT)
    connection_type = getString(connection, adress)
    
    try:
        CONN_DICT[connection_type](connection, adress)
    except KeyError:
        closeConnectionError(connection, adress)


#----------------------------------
# Main server connection accepting process
#----------------------------------        


def acceptConnections(server_socket, server_pipe):
    
    server_socket.listen(MAX_CONNECTIONS)
    server_socket.settimeout(ACCEPT_TIMEOUT)
    
    print('Starting Connections Acceptation...')
    
    while (True):
        
        if (checkPipe(server_pipe)):
            break
        
        try:
            connection, adress = server_socket.accept()
        except: 
            continue
        
        print('Connection Accepted from ADR {}.'.format(adress))
        client_process = mulpro.Process(target = serveClient,
                                        args = (connection, adress))
        client_process.daemon = False
        client_process.start()
    
    server_socket.close()
    return 1


#----------------------------------
# Function for __init__ server from ServerCommander
#----------------------------------
    

def initializeServer(host, port):

    parent_end, child_end = mulpro.Pipe()
    
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((str(host), int(port)))
    except:
        print("Can't bind to ({}, {}). Aborting server __init__.".format(host, port))
        return -1
    
    server_process = mulpro.Process(target = acceptConnections,
                                    args = (server_socket, child_end))
    server_process.daemon = False
    server_process.start()
    
    print('Server Initialized.')
    
    return parent_end


#----------------------------------
# DICTIONARY which contains references to ServeClient functions
#----------------------------------

CONN_DICT = {
        'UPDATE' : updateClient,
        'STATS': receiveStatistic
        }






