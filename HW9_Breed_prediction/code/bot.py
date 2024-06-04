import os
import torch as T
import requests
import numpy as np
from telegram import Bot
from asyncio import Queue
from telegram.ext import Updater, CommandHandler, Filters, MessageHandler
from dotenv import load_dotenv

from model.scripts.labels import LABELS, TRANSLATED_LABELS
from model.scripts.models import Model

import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models
from torchvision import transforms

from torch import nn

load_dotenv()
TOKEN = os.getenv('TG_API_TOKEN')
YaTOKEN = os.getenv('YaGPT_API_TOKEN')
#target_channel_id = 
bot = Bot(token=TOKEN)


inception = models.inception_v3()
resnet50 = models.resnet50()

# replace the last fully connected layer with a Linnear layer 133 output
in_features_resnet50 = resnet50.fc.in_features
in_features_inception = inception.fc.in_features

resnet50.fc = nn.Linear(in_features_resnet50, 120)
inception.fc = nn.Linear(in_features_inception, 120)

model = Model(inception, resnet50)
model = T.load('model/resnet-inception-sgd.pt',map_location=T.device('cpu'))

#model = torch.load('model/Resnet50_transfer_dogs.pt', map_location=torch.device('cpu'))

def start(update, context):
    update.message.reply_text('Welcome to doggy bot!')

def help_(updater, context): 
	updater.message.reply_text("Just send the image you want to classify.")

def message(updater, context):
	msg = updater.message.textx
	print(msg)
	updater.message.reply_text(msg)



def send_prompt(TOKEN, DOG):
	prompt = {
		"modelUri": "gpt://b1ge20mnmprarl42trg5/yandexgpt-lite",
		"completionOptions": {
			"stream": False,
			"temperature": 0.6,
			"maxTokens": "2000"
		},
		"messages": [
			{
				"role": "system",
				"text": "Ты известный разводчик собак различных пород. Ты знаешь все о собаках"
			},
			{
				"role": "user",
				"text": f"Привет, Разведчик! Расскажи несколько интересных фактов про породу собак {DOG}. Начинай сразу писать факт, без обращения"
			},
		]
	}


	url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
	headers = {
		"Content-Type": "application/json",
		"Authorization": f"Api-Key {TOKEN}"
	}

	response = requests.post(url, headers=headers, json=prompt)
	result = response.text
	fact = result.split('text":"')[1].split('"},')[0].replace('\\n', '')
	return fact



def image(updater, context):
	photo = updater.message.photo[-1].get_file()
	photo.download("imgs/img.jpeg")
	transform = transforms.Compose([
		transforms.Resize(size = 256),
		transforms.CenterCrop(size = 224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406],
							[0.229, 0.224, 0.225])
	])

	image = Image.open('imgs/img.jpeg')
	input_tensor = transform(image)
	input_batch = input_tensor.unsqueeze(0)

	with T.no_grad():
		output = model(input_batch)

	# predicted_class_index = T.argmax(output[0], dim=0).item()
	probabilities = T.nn.functional.softmax(output[0], dim=0)

	# Get the predicted class index
	predicted_class_index = T.argmax(probabilities).item()

	pred = LABELS[predicted_class_index]
	pred_rus = TRANSLATED_LABELS[predicted_class_index]
	print(f'English: {pred} \tRussian: {pred_rus}')

	def __round__(self, ndigits=0):
		return round(self.age, ndigits)

	#percentage = probabilities[predicted_class_index]
	#print(percentage)

	updater.message.reply_text(f'Модель определила, что с вероятностью {round(float(probabilities[predicted_class_index])*100, 2)}% это {pred_rus}')

	fact = send_prompt(YaTOKEN, pred_rus)

	updater.message.reply_text(f'Кстати, держи интересный факт об этих собаках. {fact}')



updater = Updater(bot=bot)
dispatcher = updater.dispatcher

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("help", help_))

dispatcher.add_handler(MessageHandler(Filters.text, message))

dispatcher.add_handler(MessageHandler(Filters.photo, image))

updater.start_polling()
updater.idle()
