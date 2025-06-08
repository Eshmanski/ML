import pandas
import numpy as np

emails = pandas.read_csv('../_Mock/emails.csv')

def process_email(text):
    text = text.lower()
    return list(set(text.split()))

emails['words'] = emails['text'].apply(process_email)

num_emails = len(emails)
num_spam = sum(emails['spam'])

print("Number of emails:", num_emails)
print("Number of spam emails:", num_spam)
print()

# Calculating the prior probability that an email is spam
print("Probability of spam:", num_spam/num_emails)

model = {}

for index, email in emails.iterrows():
    for word in email['words']:
        if word not in model:
            model[word] = { 'spam': 1, 'ham': 1 }
        if word in model:
            if email['spam']:
                model[word]['spam'] += 1
            else:
                model[word]['ham'] += 1

def predict_naive_bayes(email):
    total = len(emails)
    num_spam = sum(emails['spam'])
    num_hum = total - num_spam
    email = email.lower()
    words = set(email.split())
    spams = [1.0]
    hams = [1.0]
    for word in words:
        if word in model:
            spams.append(model[word]['spam'] / num_spam * total)
            hams.append(model[word]['ham'] / num_hum * total)
    prod_spams = np.prod(spams) * num_spam
    prod_hams = np.prod(hams) * num_hum
    return prod_spams / (prod_spams + prod_hams)

example_1 = "lottery sale"
example_2 = "Hi mom how are you"
example_3 = "Hi MOM how aRe yoU afdjsaklfsdhgjasdhfjklsd"
example_4 = "meet me at the lobby of the hotel at nine am"
example_5 = "enter the lottery to win three million dollars"
example_6 = "buy cheap lottery easy money now"
example_7 = "Grokking Machine Learning by Luis Serrano"
example_8 = "asdfgh"

print('example: ' + example_1)
print(predict_naive_bayes(example_1))

print('example: ' + example_2)
print(predict_naive_bayes(example_2))

print('example: ' + example_3)
print(predict_naive_bayes(example_3))

print('example: ' + example_4)
print(predict_naive_bayes(example_4))

print('example: ' + example_5)
print(predict_naive_bayes(example_5))

print('example: ' + example_6)
print(predict_naive_bayes(example_6))

print('example: ' + example_7)
print(predict_naive_bayes(example_7))

print('example: ' + example_8)
print(predict_naive_bayes(example_8))