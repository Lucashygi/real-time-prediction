# **Como realizar a inferência do seu modelo de detecção de objetos em Tempo Real** ⏳🤖👩‍💻💻

Depois que você treinou e validou o seu modelo no *Google Colab*, agora iremos validar o seu modelo realizando inferência em tempo real ou em video.

Para isso precisamos preparar o ambiente para podermos executar com sucesso o 
nosso modelo, portanto iremos divir esse tutorial nas seguintes etapas:

1. Instalação das Bibliotecas;
2. Clonagem do Repositório Tensorflow API;
3. Instalação dos pacotes do Repositório clonado; 
4. Preparação dos arquivos Necessarios;
5. Executar o script e correr para o abraço;

Nesse primeiro momento, esse tutorial se refere a Arquitetura EfficientDet;

![](https://img.shields.io/badge/Visual_Studio-5C2D91?logo=visual%20studio&logoColor=white)

## Executando as Etapas:

Antes de mais nada, abra o terminal e execute esses dois comandos `PIP`:

```
$ pip uninstall object-detection-0.1
$ pip uninstall object-detection
```

Nessa primeira etapa, iremos instalar os pacotes necessários através de um script
apenas, para isso, abra o terminal de comando na pasta raiz do projeto e 
execute o seguinte:
```
$	python script/setup.py
```
Se tudo ocorrer bem, você devera estar com `Instalador do Visual Studio` aberto,
como a representação da imagem.

<img src="images\visual-studio-installer.png" alt="model_saved" style="height: 360px; width:780px;"/>

▶ Com o instalador aberto, siga as instruções das imagens abaixo clickando nas áreas
demarcadas:

## **Passo 1**
▶ Click em modificar para abrir os pacotes disponiveis

<img src="images\visual-studio-installer-modificar.png" alt="model_saved" style="height: 360px; width:780px;"/>

## **Passo 2**
▶ Em seguida, seleciona o pacote e click em 'modificar' para começar a instalação

<img src="images\visual-studio-installer-instalacao.png" alt="model_saved" style="height: 360px; width:780px;"/>

## **Passo 3**
▶ Após o passo 2, a instalação deve começar

<img src="images\visual-studio-installer-instalacao-processo.png" alt="model_saved" style="height: 360px; width:780px;"/>


## 🤖 **EfficientDet**  
Nessa primeira etapa, iremos preparar o ambiente para a arquitetuta *EfficientDet*.

🤖 1. Primeira Etapa: Abra o terminal na raiz do projeto execute o seguinte 
comando para clonar o repositorio que contem os scrips necessários e instalar
as dependências e pacotes.
```
$	python script/setup.py
```

🤖 2. Segunda Etapa: Instalar o *protobuf releases* em seu sistema operacional,  
caso você não tenha. Para isso, deixarei um tutorial que você possar estar seguindo  
para efetuar a instalação;

[![Tutorial Youtube](https://img.shields.io/badge/youtube-red.svg?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=ES_GI-lmhEU)

🤖 3. Terceira Etapa: Instalar dependêcias via *PROTOBUF*, com o protobuf instalado
abra o terminal na raiz do projeto e navegue ate a pasta `/research`, para
isso, execute na raiz do projeto:
```
$ cd models/research
```
em seguida, execute:
```
$ protoc object_detection/protos/*.proto --python_out=.
```
por fim, efetue as ultimas instalações, executando:
```
$ python object_detection/packages/tf2/setup.py install
```
Se tudo ocorrer bem e com sucesso, você agora esta pronto para executar o script
principal.

🔥 Por fim, execute esse script no terminal no mesmo diretório que os anteriores:
```
$ SET PYTHONPATH=%cd%:%cd%\slim
```

### 🗄 **Organizando os arquivos.**

Agora que você ja preparou o Ambiente e esta tudo pronto, vamos organizar os 
arquivos necessarios, que são eles:
* 📁 *pipeline_file.config* ou *pipeline.config*
* 📁 *label_map.pbtxt*
* 📁 *(model ckpt)*
* 📁 *(weight ckpt)*

Esses 4 são os principais arquivos para a nossa inferência. Quando você terminou 
de treinar o seu modelo customizado no Beegeye, foi gerado todos essesa arquivos 
listados anteriormente. 

- 📁**pipeline_file.config ou pipeline.config** - É o arquivo que usamos para  
efetuar o treinamento, que contêm as nossas configurações personalizadas, esse  
arquivo é salvo dentro do seu *GOOGLE DRIVE*   dentro da pasta *fine-tuned_model*, 
verifique obter corretamente esse arquivo.

	<img src="images\drive-fine_tuned_model.png" alt="fine-tuned" style="height: 360px; width:780px;"/>

- 📁**label_map.pbtxt** - É o arquivo que usamos para rotular nossas classes, que 
a imagem abaixo esta representando. Esse arquivo esta dentro da pasta *data.zip*  
que o *Beegeye* gera e faz upload da mesma dentro do Google Drive.
  
	<img src="images\label-map.png" alt="label map" style="height: 360px; width:780px;"/>


- 📁**model ckpt e weight ckpt** - Aqui são os arquivos gerados pelo treinamento,  
ou seja, os checkpoints. Esses checkpoints você pode estar obtendo dentro da pasta  
*model_saved* no ambiente do *Colab*. Onde vai  estar localizado todos os 
chekpoints gerados. Cada um dos ckpts possui uma numeração no mesmo nome, por 
exemplo: `ckpt-0`, `ckpt-1`, etc. Preste atenção nos arquivos, cada numeração 
possue dois arquivos, um `.data-00000-of-00001` e `.index` e são esses arquivos 
que você tem que ter.

	<img src="images\model_saved.png" alt="model_saved" style="height: 360px; width:780px;"/>

ou você pode estar encontrando os últimos checkpoints gerados na pasta drive:

	<img src="images\checkpoint.png" alt="model_saved" style="height: 360px; width:780px;"/>

Bem, possuindo todos os arquivos listado acima, agora você deve armazerna-los  
dentro das seguintes pastas:

✔ pipeline_file.config dentro da pasta models/efficientdet

✔ label_map.pbtxt dentro da pasta models/efficientdet

✔ ckpts dentro da pasta models/efficientdet

Feito isso, abra o script `real-time-prediction.py` e verifique nas seguinte  
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

✔ inputVideo representa a variavel que carregará o video, webcam ou qualquer  
outra entrada de video, altere essa variavel.

✔ outputVideo - Gravara a predição em  formato de video.


Portanto, com essas etapas você deve conseguir com sucesso realizar  inferencia  
do seu modelo treinado através do BeegEye em Tempo real para a arquitetura **EfficientDet**


### 👀 **OBSERVAÇÃO** 👀

Você pode tentar usar esse mesmo script para as outras arquiteturas que o BeegEye  
disponibiliza, só lembre de usar os ckpts principais `ckpt-index` e `ckpt.data`.


## 📱 **Qualquer Dúvida ou Erros, entre em Contato via:**
---
[![Tutorial Youtube](https://img.shields.io/badge/WhatsApp-gree.svg?logo=WhatsApp&logoColor=white)](https://api.whatsapp.com/send?phone=5547991081602)  

[![Tutorial Youtube](https://img.shields.io/badge/Email-blue.svg?logo=microsoft-outlook&logoColor=white)](https://api.whatsapp.com/send?phone=5547991081602)
