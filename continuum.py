#Install:
#sudp apt update
#sudo apt-get install ssmtp 
#sudo apt-get install mailutils
#sudo apt install python3-pip
#pip3 install mailutils
#pip3 install bs4
#Run:
#screen
#python3 continuum.py

from glob import glob
from urllib.request import urlopen
from bs4 import BeautifulSoup
import time 
import smtplib
import datetime

smtpUser = 'govopoly@gmail.com'
smtpPass = ''

toAdd_lst = ['jjj4se@virginia.edu', 'gjabbour@gmail.com','lydiajabbour@gmail.com']
toAdd_admin = 'jjj4se@virginia.edu'
fromAdd = smtpUser

url = 'https://continuumallston.com/live/floor-plans-pricing'
apartments_types_looking_for = ['Studio','1 Bedroom','2 Bedroom']
max_price_looking_for = 3000

update_num = 1
apartments_emailed = {} #{'apart_num': price}

previous_day = -1

def get_apartment_prices():
    '''Read URL Page and check for new apartments'''

    global apartments_emailed, update_num

    message = '\n'
    send_msg = False

    try:
        #Read Page as text
        page = urlopen(url).read()
        #Initialize Beautiful Soup
        soup = BeautifulSoup(page, features="html.parser")
        #Find attributes
        all_apartments_numbers = [item["data-id"] for item in soup.find_all() if "data-id" in item.attrs]
        all_apartment_prices = [float(item["data-price"]) for item in soup.find_all() if "data-price" in item.attrs]
        all_apartment_types = [item["data-type"] for item in soup.find_all() if "data-type" in item.attrs]

        #Sanity Check
        assert len(all_apartments_numbers) == len(all_apartment_prices) == len(all_apartment_types), 'Lengths of Readings not equal'

        #Read through all listings
        for i, apartment_type in enumerate(all_apartment_types):
            #Check if apartment fits criteria
            if (apartment_type in apartments_types_looking_for) and (all_apartment_prices[i] <= max_price_looking_for):
                #Check if already sent email about this apartment
                if all_apartments_numbers[i] in apartments_emailed:
                    #If already sent, check if price has decreased
                    if (all_apartment_prices[i] < apartments_emailed[all_apartments_numbers[i]]):
                        #Update apartments_called
                        apartments_emailed[all_apartments_numbers[i]] = all_apartment_prices[i]
                        #add message
                        message += 'PRICE Decreased! Apartment number: ' + all_apartments_numbers[i] + ' ('+apartment_type +') ' + 'price: $'+ str(all_apartment_prices[i]) + '\n'
                        send_msg = True

                #Haven't emailed this apartment and is under max price, then email
                else:
                    #Update apartments_called
                    apartments_emailed[all_apartments_numbers[i]] = all_apartment_prices[i]
                    #add message
                    message += 'NEW Listing! Apartment number: ' + all_apartments_numbers[i] + ' ('+apartment_type +') ' + 'price: $'+ str(all_apartment_prices[i]) + '\n'
                    send_msg = True

        if send_msg:
            message+="View here: https://continuumallston.com/live/floor-plans-pricing"
            #Send all messages in one email
            for email_addy in toAdd_lst:
                subject = 'Continuum Apartment Update #' + str(update_num)
                send_email_update(message, email_addy, subject)
                print("Emails Sent")
            
            update_num+=1
        else:
            print("No new update")
    except Exception as e:
        #Send error to admin
        send_error_email(str(e), 'get_apartment_prices()')
        print('Error in get_apartment_prices(): ' + str(e))        

def send_email_update(msg, email_addy, subject): 
    toAdd = email_addy
    header = 'To: ' + toAdd + '\n' + 'From: ' + fromAdd + '\n' + 'Subject: ' + subject
    body = msg

    s = smtplib.SMTP('smtp.gmail.com',587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login(smtpUser, smtpPass)

    s.sendmail(fromAdd, toAdd, header + '\n\n' + body)
    s.quit()

def weekly_status_email():
    '''Send an email to the admin every Sunday
       to verify this script is still running
    '''
    global previous_day

    try:
        #Returns 0-6 where 0 is monday and 6 is Sunday
        day = datetime.datetime.today().weekday()

        #Check if day is Sunday and we havent already sent an email today
        if (not previous_day == day) and (day == 6):
            subject = '[Status Check] Continuum.py Script'
            msg = '>>> Continuum.py script running <<<'
            send_email_update(msg, toAdd_admin, subject)
        
        #update the day today
        previous_day = day

    except Exception as e:
        #Send error to admin
        send_error_email(str(e), 'weekly_status_email()')
        print('Error in weekly_status_email(): ' + str(e))

def send_error_email(error_message, function_of_error):
    '''Send error message to admin
    
    Args:
        error_message: A string version of the exception
        function_of_error: name of the function which the error was found.
    '''
    #Construct Email Message to include function of error and error message
    error_message = f'Continuum.py script error in {function_of_error}: ' + str(error_message)
    #Create the Subject
    subject = '[ERROR] Continuum.py script'
    #Send the email to the admin
    send_email_update(error_message, toAdd_admin, subject)
    #Print to console
    print("Error email sent")

def start_sending_emails():
    '''Runner for sending emails'''

    while True:
        # Apartment Prices
        get_apartment_prices()
        # Status Check
        weekly_status_email()
        
        print("Sleeping for 30 Minutes")
        time.sleep(1800) #30 Minutes


if __name__ == "__main__":

    start_sending_emails()
