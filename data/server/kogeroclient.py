"""
    KOGERO TEAM
    (!!!) Warning: USE ONLY updateClient & sendStatistic functions!
"""

import sys
import threading
import queue
import socket
import pickle

PACKET_SIZE = 4096
CURRENT_VERSION = '0.1'
RECV_TIMEOUT = 10
SEND_TIMEOUT = 10
STR_ENCODING = 'UTF-8'


#----------------------------------
# Function to establish connection with server
#----------------------------------


def establishConnection(host, port):
    
    connection_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection_socket.settimeout(10)
    try:
        connection_socket.connect((host, port))
    except:
        sys.exit(0)
    return connection_socket


#----------------------------------
# Functions for sending data
#----------------------------------


def sendString(connection, string, timeout = SEND_TIMEOUT):
    
    encoded_string = bytes(string, STR_ENCODING)
    sendRaw(connection, encoded_string, timeout)
    return 1

def sendRaw(connection, bytes_stream, timeout = SEND_TIMEOUT):
    
    connection.settimeout(timeout)
    try:
        connection.sendall(bytes_stream)
    except:
        connection.close()
        sys.exit(0)
    return 1


#----------------------------------
# Functions for receiving data
#----------------------------------


def getString(connection, timeout = RECV_TIMEOUT):
    
    encoded_string = getRaw(connection, timeout)
    return str(encoded_string, STR_ENCODING)

def getRaw(connection, timeout = RECV_TIMEOUT):
      
    data = b''
    connection.settimeout(timeout)
    while(True):
        try:
            fragment = connection.recv(PACKET_SIZE)
        except:
            connection.close()
            sys.exit(0)
        data += fragment
        if (len(fragment) < PACKET_SIZE):
            break
    if (data == b''):
        connection.close()
        sys.exit(0)
    return data


#----------------------------------
# Fucntions for handling types of connections in specialized Thread
#----------------------------------


def sendStatisticThread(host, port, form_array, emotions_array, result_queue):
    
    connection_socket = establishConnection(host, port)
    sendString(connection_socket, 'STATS')
    answer = getString(connection_socket)
    if (answer == 'READY'):
        send_dict = {
                'form' : form_array,
                'emotions' : emotions_array
                }
        pickled_dict = pickle.dumps(send_dict)
        sendRaw(connection_socket, pickled_dict)
        answer = getString(connection_socket)
        if (answer == 'END'):
            result_queue.get()
            result_queue.put(1)
    connection_socket.close()

def updateClientThread(host, port, current_version, result_queue):
    
    pickled_data = None
    unpickled_dictionary = None
    matrix_7_16 = None
    matrix_16_12 = None
    connection_socket = establishConnection(host, port)
    sendString(connection_socket, 'UPDATE')
    server_version = getString(connection_socket)
    if (float(server_version) > float(current_version)):
        sendString(connection_socket, 'NEXT')
        answer = getString(connection_socket)
        if (answer != 'READY'):
            connection_socket.close()
            sys.exit(0)
        else:
            sendString(connection_socket, 'READY')
            pickled_data = getRaw(connection_socket)
            sendString(connection_socket, 'END')
    else:
        sendString(connection_socket, 'END')
        result_queue.get()
        result_queue.put(2)
        sys.exit(0)
    connection_socket.close()
    
    try:
        unpickled_dictionary = pickle.loads(pickled_data)
        matrix_7_16 = unpickled_dictionary['matrix_7_16']
        matrix_16_12 = unpickled_dictionary['matrix_16_11']
        bias_1 = unpickled_dictionary['bias_1']
        bias_2 = unpickled_dictionary['bias_2']
    except:
        sys.exit(0)


    result_queue.get()
    result_queue.put(1)
    result_queue.put(matrix_7_16)
    result_queue.put(matrix_16_12)
    result_queue.put(bias_1)
    result_queue.put(bias_2)
    result_queue.put(str(server_version))
    
    
#----------------------------------
# Functions for calling <SendFunction>Thread
#----------------------------------


def sendStatistic(host, port, form_array, emotions_array):
    
    result_queue = queue.Queue()
    result_queue.put(0)
    connection_thread = threading.Thread(target = sendStatisticThread,
                                         args = (host, port, form_array,
                                                 emotions_array, result_queue))
    connection_thread.start()
    connection_thread.join()
    
    send_status = result_queue.get()
    return send_status
    
def updateClient(host, port, current_version):
    
    matrix_7_16 = None
    matrix_16_12 = None
    bias_1 = None
    bias_2 = None
    new_version = current_version
    result_queue = queue.Queue()
    result_queue.put(0)
    connection_thread = threading.Thread(target = updateClientThread,
                                         args = (host, port,
                                                 current_version, result_queue))
    connection_thread.start()
    connection_thread.join()
    
    update_status = result_queue.get()
    if (update_status == 1):
        try:
            matrix_7_16 = result_queue.get()
            matrix_16_12 = result_queue.get()
            bias_1 = result_queue.get()
            bias_2 = result_queue.get()
            new_version = result_queue.get()
        except:
            pass
    return update_status, matrix_7_16, matrix_16_12, bias_1, bias_2, new_version
    

if __name__ == '__main__':
	input()
	res = sendStatistic('10.97.91.153', 64000, [1, 2], [3, 4])
	print(res)
	input()