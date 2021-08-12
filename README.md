# **Como realizar a inferência do seu modelo de detecção de objetos em Tempo Real** ⏳🤖👩‍💻💻

Depois que você treinou e validou o seu modelo no *Google Colab*, agora iremos validar o seu modelo realizando inferência em tempo real ou em video.

Para isso precisamos preparar o ambiente para podermos executar com sucesso o 
nosso modelo, vamos dividir as etapas por tipo de arquitetura que foi treinado
o modelo.  

## 🤖 **EfficientDet**  
Nessa primeira etapa, iremos preparar o ambiente para a arquitetuta *EfficientDet*.

🤖 1. Primeira Etapa: Abra o terminal na raiz do projeto execute o seguinte</br>
comando para clonar o repositorio que contem os scrips necessários.<br/>
```
$	python script/clone.py
```

🤖 2. Segunda Etapa: Instalar o *protobuf releases* em seu sistema operacional, </br>
caso você não tenha. Para isso, deixarei um tutorial que você possar estar seguindo </br>
para efetuar a instalação;<br/>

[![Tutorial Youtube](https://img.shields.io/badge/youtube-red.svg?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=ES_GI-lmhEU)

🤖 3. Terceira Etapa: Instalar dependecias via *PROTOBUF*, com o protobuf instalado
abra o terminal na raiz do projeto e execute:
```
$ protoc models/research/object_detection/protos/*.proto --python_out=.
```
em seguida, execute:
```
$ python models/research/object_detection/packages/tf2/setup.py install
```
Se tudo ocorrer bem e com sucesso, você agora esta pronto para executar o script</br>
principal.

### 🗄 **Organizando os arquivos.**

Agora que você ja preparou o Ambiente e esta tudo pronto, vamos organizar os </br>
arquivos necessarios, que são eles:
* 📁 *pipeline_file.config* ou *pipeline.config*
* 📁 *label_map.pbtxt*
* 📁 *(model ckpt)*
* 📁 *(weight ckpt)*

Esses 4 são os principais arquivos para a nossa inferência. Quando você terminou</br>
de treinar o seu modelo customizado no Beegeye, foi gerado todos essesa arquivos</br>
listados anteriormente. 

- 📁**pipeline_file.config ou pipeline.config** - É o arquivo que usamos para </br>
efetuar o treinamento, que contêm as nossas configurações personalizadas, esse </br>
arquivo é salvo dentro do seu *GOOGLE DRIVE* </br> dentro da pasta *fine-tuned_model*,</br>
verifique obter corretamente esse arquivo.

	<img src="images\drive-fine_tuned_model.png" alt="fine-tuned" style="height: 360px; width:780px;"/>

- 📁**label_map.pbtxt** - É o arquivo que usamos para rotular nossas classes, que</br>
a imagem abaixo esta representando. Esse arquivo esta dentro da pasta *data.zip* </br>
que o *Beegeye* gera e faz upload da mesma dentro do Google Drive.
  
	<img src="images\label-map.png" alt="label map" style="height: 360px; width:780px;"/>


- 📁**model ckpt e weight ckpt** - Aqui são os arquivos gerados pelo treinamento, </br>
ou seja, os checkpoints. Esses checkpoints você pode estar obtendo dentro da pasta </br>
*model_saved* no ambiente do *Colab*. Onde vai </br>estar localizado todos os</br>
chekpoints gerados. Cada um dos ckpts possui uma numeração no mesmo nome, por</br>
exemplo: `ckpt-0`, `ckpt-1`, etc. Preste atenção nos arquivos, cada numeração</br>
possue dois arquivos, um `.data-00000-of-00001` e `.index` e são esses arquivos</br>
que você tem que ter.

	<img src="images\model_saved.png" alt="model_saved" style="height: 360px; width:780px;"/>

ou você pode estar encontrando os últimos checkpoints gerados na pasta drive:

	<img src="images\checkpoint.png" alt="model_saved" style="height: 360px; width:780px;"/>

Bem, possuindo todos os arquivos listado acima, agora você deve armazerna-los </br>
dentro das seguintes pastas:

✔ pipeline_file.config dentro da pasta models/efficientdet

✔ label_map.pbtxt dentro da pasta models/efficientdet

✔ ckpts dentro da pasta models/efficientdet

Feito isso, abra o script `real-time-prediction.py` e verifique nas seguinte </br>
linhas 46 ao 51 as seguintes variaveis.

~~~python
46 pipeline_config  = './models/efficientdet/pipeline_file.config'
47 label_map_path   = './models/efficientdet/label_map.pbtxt'
# Altere o numero do ckpt para o numero que corresponde ao seu chekpoint
48 model_dir 	    = './models/efficientdet/ckpt-2' 
49 checkpoint	    = './models/efficientdet/ckpt-2'

50 inputVideo  	    = './videos/input.mp4' # or 0 to started Webcan
51 outputVideo      = './output.mp4'
~~~

Verifique as variaveis e o caminho que corresponde a cada um dos arquivos. 

### 🚨 **Atenção - As seguite variaveis** 🚨 

✔ inputVideo representa a variavel que carregará o video, webcam ou qualquer </br>
outra entrada de video, altere essa variavel.

✔ outputVideo - Gravara a predição em  formato de video.


Portanto, com essas etapas você deve conseguir com sucesso realizar  inferencia </br>
do seu modelo treinado através do BeegEye em Tempo real para a arquitetura **EfficientDet**


### 👀 **OBSERVAÇÃO** 👀

Você pode tentar usar esse mesmo script para as outras arquiteturas que o BeegEye </br>
disponibiliza, só lembre de usar os ckpts principais `ckpt-index` e `ckpt.data`.


## 📱 **Qualquer Dúvida ou Erros, entre em Contato via:**
---
[![Tutorial Youtube](https://img.shields.io/badge/WhatsApp-gree.svg?logo=WhatsApp&logoColor=white)](https://api.whatsapp.com/send?phone=5547991081602)  

[![Tutorial Youtube](https://img.shields.io/badge/Email-blue.svg?logo=microsoft-outlook&logoColor=white)](https://api.whatsapp.com/send?phone=5547991081602)
