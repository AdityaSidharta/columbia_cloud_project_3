import boto3
import string
import sys
import numpy as np
from hashlib import md5
from sagemaker.mxnet.model import MXNetPredictor
import mailparser
import json
import time
import urllib.parse

vocabulary_length = 9013
model = MXNetPredictor('sms-spam-classifier-mxnet-2022-11-21-22-42-04-356')
s3 = boto3.resource('s3')
ses = boto3.client("ses")

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

def vectorize_sequences(sequences, vocabulary_length):
    results = np.zeros((len(sequences), vocabulary_length))
    for i, sequence in enumerate(sequences):
       results[i, sequence] = 1. 
    return results

def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]

def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
    return hashing_trick(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)


def hashing_trick(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]

def get_mail_boundary(mail_bodys):
    result = len(mail_bodys)
    for i in range(result):
        if '--- mail_boundary ---' in mail_bodys[i]:
            return i
    return result

def get_body(mail_bodys):
    result = []
    for line in mail_bodys:
        line = line.strip()
        if line != '':
            result.append(line)
    return ' '.join(result)[:240]

def predict(body):
    one_hot_test_messages = one_hot_encode([body], vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
    result = model.predict(encoded_test_messages)
    predicted_label = result['predicted_label'][0][0]
    predicted_probability = result['predicted_probability'][0][0]
    if predicted_label == 1:
        return 'SPAM',(predicted_probability * 100.)
    else:
        return 'HAM', (1. - predicted_probability) * 100.

def get_response_title(received_date):
    return "Spam Classification for email on {}".format(received_date)

def get_response_body(received_date, subject, body, classification, classification_score):
    return """We received your email sent at {receive_date} with the subject {email_subject}
    
    Here is a 240 character sample of the email body: {email_body}
    
    The email was categorized as {classification} with a {classification_score}% confidence.
    """.format(
        receive_date= received_date,
        email_subject= subject,
        email_body= body,
        classification= classification,
        classification_score= round(classification_score,2))

def verify_email(email_address):
    response = ses.verify_email_identity(
        EmailAddress=email_address
    )
    return response

def send_email(email_address, response_title, response_body):
    CHARSET = "UTF-8"
    response = ses.send_email(
        Destination={
            "ToAddresses": [
                email_address,
            ],
        },
        Message={
            "Body": {
                "Text": {
                    "Charset": CHARSET,
                    "Data": response_body,
                }
            },
            "Subject": {
                "Charset": CHARSET,
                "Data": response_title,
            },
        },
        Source="aditya@sidharta.club",
    )

def manage_email(email_address, response_title, response_body):
    retry = 5
    while retry:
        try:
            verify_email(email_address)
            print("Verify Email : {}".format(email_address))
            time.sleep(10)
            send_email(email_address, response_title, response_body)
            print("Send Email : {}".format(email_address))
            break
        except Exception as e:
            print(e)
            if retry == 0:
                raise ValueError("Send email has failed")
            retry = retry - 1
            print("retry {} left".format(retry))
            continue
    return

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event))
    try:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
        obj = s3.Object(bucket, key)
        print("Key : {}, Object {}".format(key, obj))
        value = obj.get()['Body'].read().decode('utf-8')
        mail = mailparser.parse_from_string(value)
        mail_bodys = mail.body.split('\n')
        mail_bodys = mail_bodys[:get_mail_boundary(mail_bodys)]
        body = get_body(mail_bodys)
        subject = mail.subject
        received_date = mail.date.isoformat()
        classification, classification_score = predict(body)
        print("Body : {}, Subject : {}, Received Date : {}, Classification : {}, Classification Score : {}".format(body, subject, classification, classification_score, received_date))
        email_address = mail.from_[0][1]
        response_title = get_response_title(received_date)
        response_body = get_response_body(received_date, subject, body, classification, classification_score)
        print("Response title : {}, Response Body : {}".format(response_title, response_body))
        manage_email(email_address, response_title, response_body)
    except Exception as e:
        print(e)
        print('Error processing object {} from bucket {}.'.format(key, bucket))
        raise e