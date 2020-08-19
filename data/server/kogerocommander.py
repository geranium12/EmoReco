"""
    KOGERO TEAM
"""
import kogeroserver

# startserver HOST PORT 
# closeserver
#
# startserver 192.168.0.116 60606

        
IS_ACTIVE = False

def getCommand(message):
    fragmented_message = list(message.split(' '))
    return fragmented_message
    
if __name__ == '__main__':
    
    command_pipe = None
    
    while True:
        message = input()
        command = getCommand(message)
        
        if (command[0] == 'startserver' and IS_ACTIVE == False):
            command_pipe = kogeroserver.initializeServer(command[1], command[2])
            if (command_pipe != -1):
                IS_ACTIVE = True
        elif (command[0] == 'closeserver' and IS_ACTIVE == True):
            kogeroserver.closeServer(command_pipe)
            IS_ACTIVE = False
        else:
            print('Invalid commmand.')
        
        