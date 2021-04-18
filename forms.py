from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, BooleanField, SubmitField,FieldList,FormField,SelectField,validators
from wtforms.validators import DataRequired
from wtforms.widgets import html5 as h5widgets
from env_data import getEnvNames#OJO ESTO ESTÁ DUPLICADO, LIMPIARLO EN EL FUTURO

#https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-iii-web-forms

env_names = getEnvNames()
algo_names = ["DQN","PPO","A2C","TRPG","DDPG"]
policies = ["MlpPolicy", #Policy object that implements actor critic, using a MLP (2 layers of 64)
"MlpLstmPolicy",#Policy object that implements actor critic, using LSTMs with a MLP feature extraction

"MlpLnLstmPolicy",#Policy object that implements actor critic, using a layer normalized LSTMs with a MLP feature extraction

"CnnPolicy",#Policy object that implements actor critic, using a CNN (the nature CNN)

"CnnLstmPolicy",#Policy object that implements actor critic, using LSTMs with a CNN feature extraction

"CnnLnLstmPolicy",#Policy object that implements actor critic, using a layer normalized LSTMs with a CNN feature extraction

]

class singleAlgo(FlaskForm):
    algo = BooleanField("")

class singleEnv(FlaskForm):
    env = BooleanField("")
class singleEnv(FlaskForm):
    env = BooleanField("")

 
class envForm(FlaskForm):#TODO:BORRAR ESTE Y EL DE ABAJO SI FINALMENTE NO LO USO
    envs = FieldList(FormField(singleEnv),min_entries=len(env_names))
    submit = SubmitField('Submit')


class algosForm(FlaskForm):
    algos = FieldList(FormField(singleAlgo),min_entries=len(algo_names))
    submit = SubmitField('Submit')


class configForm(FlaskForm):
    algos = FieldList(FormField(singleAlgo),min_entries=len(algo_names))
    envs = FieldList(FormField(singleEnv),min_entries=len(env_names))

  
   # policy = SelectField(u'policy', choices=[(i,pol) for i,pol in enumerate(policies)])#we assume same policy for fair comparison
    experiment_name = StringField(u'Experiment Name', [validators.required(), validators.length(max=15)])
    policy = SelectField(u'policy', choices=[pol for pol in policies],validators=[validators.required()])#we assume same policy for fair comparison
    n_timesteps = IntegerField(widget=h5widgets.NumberInput(),validators=[validators.required()])#we assume same number for fair comparison
    
    submit = SubmitField('Submit')
#UNIFICAR Y PONER TODA LA CONFIGURACIÓN EN UN FORM

#HACER FORMULARIO DE LAS POLITICAS Y LOS TIMESTEPS
