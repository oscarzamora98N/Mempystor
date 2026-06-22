# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 10:37:28 2026

@author: osz98N
"""
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np

import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
from kneed import KneeLocator
from scipy.signal import savgol_filter
import os

# --- Optimizaciones de Matplotlib para acelar el plotting--- 
mpl.style.use('fast')
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 1.0
mpl.rcParams['agg.path.chunksize'] = 10000
# ---------------------------------------------------
mpl.rcParams['savefig.dpi'] = 600 #esto para que se guarade a alta calidad las graficas de pruebas que salen en ventana (en caso de quererlo)

from sklearn.preprocessing import RobustScaler


def leer_enJ (archivo,tecnica="CV",hasta="todo",desde=1,D=0.2): 
    """ 
    osz98N
    
    Esta función lee los archivos exportados de EC-Lab (programa del potencioestato) en el formato ".mpt"
    En la exportación hay que emplear absolute time para que no falle el Ec-Lab
    
    Diámetro del contacto de oro por defecto D=0.2cm !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Técnica por defecto CV
    
    archivo: ruta al archivo desde el directorio raiz (donde esta instalado Python)
    tecnica: técnica de medida usada en el potencioestato: CV (cyclic voltametry) o SPEIS (Staircase Potentio Electrochemical Impedance Spectroscopy)
    hasta: hasta que número de run del mismo experimento se quiere leer, por defecto ="todo" se lee todo lo que haya en la carpeta con esa flag como nombre
    desde: desde que archivo se quiere leer, por defecto desde 1, es decir el principio. Este parametro se puede usar para leer solo un archivo en una posición concreta
    """ 
    print("Leer_enJ V1.0 osz98N \n" + "\033[1;33m Semi-Auto\033[0m"+ "\033[1;34m    Mode:"+tecnica +"\033[0m") 
 #como usamos CV y SPEIS he puesto aqui una selección rápida, ya que los números de columnas, nombres y la cabecera inicial cambian

    if tecnica=="SPEIS":
        skipeables=56 #valor por defecto
        nombres=["Freq","Z_re","Z_-im","Z","Phase(Z)","Time","Ewe","I"]
        columnas_csv=[0,1,2,3,4,5,6,7]
    elif tecnica=="CV":
        skipeables=48 #la primera fila es justo donde arranca, tiene datos que no valen, skipeo de 48 (lo que pone en el ,mpt)
        nombres=["Ewe","I","nº ciclo"]
        columnas_csv=[7,8,9]
    elif tecnica=="Z fit SPEIS":
         skipeables=11
         nombres=["R2","Q3","a3","nº ciclo","Ewe","I"]
         columnas_csv=[2,3,4,6,7,8]
    else:
        print("Error tipo no reconocido")
        
 #conversión de todo a -1, asi es mas intuitivo de usar, si es -1 el while se ejecuta hasta que se rompa
    if hasta == "todo":
        hasta=-1
        
 #se incluye un desde, asi se puede leer solo un archivo concreto 
 
    temp=archivo  
    if desde<=9:
        archivo=archivo.replace("_0"+str(1)+".","_0"+str(desde)+".",1)
               
    else:
        archivo=archivo.replace("_0"+str(1)+".","_"+str(desde)+".",1)
        
    if temp==archivo and desde!=1: # Si el replace no ha hecho nada es que no ha encontrado el formato definido en el nombre y no se puede leer, por defecto el replace no raisea un errror si eso pasa
        raise FileNotFoundError(archivo+"Format not found in file name, be sure that is named ...something_XX.mpt ; X is a number")
    else: 
        pass
        
    i=desde #no es necesario pero me gusta mas el while con la i      
 #---------------------------------------------------------------------       
    while (i<hasta+1 or hasta== -1):#se lee todos los archivos flageados con ese nombre hasta donde se quiera      
 #lectura del archivo mpt: !!separador decimal ",", las 56 primeras filas son basura, he overaideado los nombres!!
        
        try:#si ya no hay mas que se leer se sale del bucle
            #Detección automática del tamaño del header que el EC lab a veces cambia el numero de skipeables en SPEIS, lo de arriba se ha quedado por si esto falla
            with open(archivo, 'r', encoding="latin1") as f:#apertura temporal asi es mas compacto 
                f.readline() # EC-Lab ASCII FILE, lee la primera fila que no interesa 
                line2 = f.readline() #Ahora lee la segunda que si intersa 
                if "Nb header lines" in line2:
                    skipeables = int(line2.split(":")[1].strip()) - 1 #Para extraer el numero 
            # --------------------------------------------------
            dftemp=pd.read_csv(archivo, sep="\t",skiprows=skipeables,decimal=',',header=0,names=nombres,usecols=columnas_csv,encoding="latin1")# encoding="latin1" el potencioestato es tan viejo que no usa utf-8, usa cosas de europa occidental con acentos y caracteres no internacionales
        except FileNotFoundError as mesg:#si no hay archivo a tomar por culo no lee mas 
            print(mesg)
            break
        
        try:#si es la primera lectura el try catch definira df con el temporal. 
            df=pd.concat([df,dftemp],axis=0)
        except NameError:
            df=dftemp
            
        temp=archivo #guardamos esto para raisear un error si no encuentra el nombre el replace (ya que por defecto si eso ocurre no tira error)
        
        #pasa al siguiente archivo numerado con ese nombre (siguiente ciclo de medidas)
        if i<9:
             archivo=archivo.replace("_0"+str(i)+".","_0"+str(i+1)+".",1)
            
        
        elif i==9:
            archivo=archivo.replace("_09.", "_10.",1)
            
       
        else:
            archivo=archivo.replace("_"+str(i)+".","_"+str(i+1)+".",1)
        i=i+1
        
    if 'df' not in locals():
        print(f"Error: No se pudo cargar ningún dato de {archivo}")
        return None
        
 #conversión  a J-----------------------------------------------------------------
    #D=0.2  #diametro del contacto de oro en cm = 0.2cm por defecto, error en D 0.01mm
    A=np.pi*(D/2)**2 #area del contacto de oro en cm^2
    #df["I"]=df["I"]/1000 #Se pasa a Amperios para tener todo en S.I.!!!!!
    df["J"]=df["I"]/A #J=I/A 

 #Textito chulo 
    print( "\033[1;35m"+ str(df.shape) +" Data points has been loaded" + " on " +tecnica + " Mode" + "\033[0m\n")    
    
 #---------------------------------------------------------------------------------------
    return (df)

def JV (df,titulo="Curva J-V",nciclo="todo",log=False, dpi=600, new_fig=True):#Dpi por defecto aunque me gusta 2000, con 600 vale para analisis normales en el lab
   """
    osz98N
    
    Esta función hace la gráfica J-V de un dataframe 
    
    df: Dataframe de pandas donde estan los datos del potencioestato y la columna con la J (véase leer_enJ)
    titulo: Titulo de la gráfica (que será el nombre el archivo tambíen), por defecto "Curva J-V: 1poroso"
    Precaución cuando se trabaje con multiporosos si se usa el por defecto
    nciclo: número del ciclo a leer, por ahora solo puede ir ciclo a ciclo en los archivos medidos por CV
    
     """ 


   if new_fig:
       plt.figure()#Para que no siga el hold

   #Escala normal/logy--------------------------------------------------------------------------------
   if log==True: #Para poder tener escala normal y logaritmica de forma compacta 
        plt.yscale("log") #Escala log en y
        df["J"]=abs(df["J"])
   else:
        pass
   #Ciclos a graficar uno/todos--------------------------------------------------------------------------------
   
   if nciclo!="todo": #grafica solo un ciclo (solo CV)
   
       plt.plot(df["Ewe"][df["nº ciclo"]==nciclo],df["J"][df["nº ciclo"]==nciclo], linewidth=2, markersize=1, label=titulo) # SPEIS no tiene n ciclo!!!!!!!!!!!!!!
       
   else: #graficar todo  (para SPEIS solo funciona este) 
       
       plt.plot(df["Ewe"],df["J"], linewidth=0.8, color="gray",zorder=1,alpha=0.7) #todo los ciclos en gris 
       
       try:    #si no existe n ciclo peta, lo que pasa cuando no hay ciclos
            ciclomin=df["nº ciclo"].min()
            ciclomax=df["nº ciclo"].max() 
            try: #Para ver la dirección del lazo una vez etiquetados, asi se puede usar antes sin problema, se pone como en los papers el primer y ultimo ciclo remarcados, lo pinta dos veces (gris y en color) pero prefiero ahorrarme el if 
           
                plt.scatter(df["Ewe"][(df["nº ciclo"]==ciclomin) & (df["State"]==0)],df["J"][(df["nº ciclo"]==ciclomin) & (df["State"]==0)],s=2,color="tab:blue",edgecolors="none",zorder=2,label="Ciclo " + str(ciclomin) + ": creciente")
                plt.scatter(df["Ewe"][(df["nº ciclo"]==ciclomin) & (df["State"]==1)],df["J"][(df["nº ciclo"]==ciclomin) & (df["State"]==1)],s=2, color="tab:orange",edgecolors="none",zorder=2,label="Ciclo " + str(ciclomin)+": decreciente")
    
                plt.scatter(df["Ewe"][(df["nº ciclo"]==ciclomax) & (df["State"]==0)],df["J"][(df["nº ciclo"]==ciclomax) & (df["State"]==0)],s=2,color="mediumorchid",edgecolors="none",zorder=3,label="Ciclo " + str(ciclomax) + ": creciente")
                plt.scatter(df["Ewe"][(df["nº ciclo"]==ciclomax) & (df["State"]==1)],df["J"][(df["nº ciclo"]==ciclomax) & (df["State"]==1)],s=2, color="mediumseagreen",edgecolors="none",zorder=3,label="Ciclo " + str(ciclomax) +": decreciente")
           
            except Exception:
                plt.plot(df["Ewe"][df["nº ciclo"]== ciclomin],df["J"][df["nº ciclo"]== ciclomin], linewidth=2, color="tab:blue", zorder=2,label="Primer ciclo")
                plt.plot(df["Ewe"][df["nº ciclo"]==ciclomax],df["J"][df["nº ciclo"]==ciclomax], linewidth=2, color="tab:orange", zorder=3,label="Último ciclo")   
       except:
            pass   
    
   #Info de la grafica y otros -------------------------------------------------------------------------------- 
   
   plt.title(titulo)
   plt.xlabel(r"V [V]")
   plt.ylabel(r"J [mA/cm$^2$]") if log==False else plt.ylabel(r"Log |J| [mA/cm$^2$]")#Para que salga el log o no
   
   plt.grid()  
   plt.legend(markerscale=5)
   
   
   plt.savefig(dpi=dpi,fname="TFM/resultados/"+titulo+".png")
   plt.show(block=False)#block=False es para que no se pare en la grafica y siga el programa corriendo, en spyder eso no ocurre, pero en antigravity si
   
   return()

def labelmem (df,describe=True,evolplot=False,sens=1,thlitoff=0.1,thhard=2,thswitch=1e-4):
   """ 
    osz98N
    
    Esta función busca los puntos de Set y Reset del memristor por definición matemática (en set/reset hay un max/min local en corriente eléctrica I)
    Considere que si el memristor falla el ciclo de lectura escritura debe eliminar ese ciclo manualmente df.drop()
    
   df: Pandas dataframe a computar, obligatorio contar con columna etiquetadora de ciclos (solo modo CV por ahora)
   describe: Descripción estadística de los ciclos on/off
   sens: Sensibilidad del algoritmo (KneeLocator)
   thlitoff: Threshold de despegue soft switching, X*(Delta entre codo y fondo)
   thhard: Threshold de pico hard switching, X*Imax(suavizada)
   thswitch: Corriente mínima absoluta para validar el set
   
   Ewe-+: Ewe max min de barrido 
   I/J+- p: I max min de barrido en el pico (hay desfase entre V e I)
   """ 
   print("Labelmem V1.0 osz98N \n" + "\033[1;33m Full Auto\033[0m"+ "\033[1;34m    Mode:" +"\033[0m") 
   
   #Detector de estado de voltametría (potenciacion=0 / depresión=1) lo que permite evitar tener dos curvas juntas cuando se ajuste en tramos de voltaje 
 #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
   #Esto permite extraer los ciclos de potenciación, al ser memristor el pico de I va desfasado de Ewe+- por eso se detecta por maximo, ese máximo de I separa potenciación de depresión 
   dfe=(pd.concat([df.groupby(by=["nº ciclo"]).max(),df.groupby(by=["nº ciclo"]).min()],axis=1, ignore_index=True)).set_axis(["Ewe+","I+ p","J+ p","Ewe-","I- p","J- p"],axis=1) #Pegrilo, puntos de set y reset computados por definición matemática (máximos y mínimos absolutos en el lazo)
   
   #Filtro que restringe la busqueda de max/min I al entorno próximo del Ewe+ / Ewe- (0.5V al rededor), asi se evita que detecte antes de tiempo el cambio de potenciacion 
   dfei=(pd.concat([df.groupby(by=["nº ciclo"]).apply(lambda g: g[g["Ewe"] >= g["Ewe"].max() - 0.5]["I"].idxmax(), include_groups=False), df.groupby(by=["nº ciclo"]).apply(lambda g: g[g["Ewe"] <= g["Ewe"].min() + 0.5]["I"].idxmin(), include_groups=False)], axis=1)).set_axis(["& I+ p","& I- p"],axis=1) #Para tener arriba la matriz limpia, se hace una separada de indirecciones (realmente posicion en index de df1) de los valores de I que interesan
   
   dfe.insert(1 ,"Ewe+ p",(((df[["Ewe","nº ciclo"]]).iloc[dfei["& I+ p"].values]).set_index("nº ciclo")).set_axis(["Ewe+ p"],axis=1))
   dfe.insert(5 ,"Ewe- p",(((df[["Ewe","nº ciclo"]]).iloc[dfei["& I- p"].values]).set_index("nº ciclo")).set_axis(["Ewe- p"],axis=1))
   
   #Es imposible hacer en Pandas una comparación (una col) valor por valor de un dataframe a otro (con diferente longitud) tambien valor por valor, es decir revisar si un valor pertenece a un rango de valores dados por dos columnas. (Es posible pero con una cosa que te puede reventar la memoria del PC) 
   #Esto en SQL se llama Non-equi Join (eso dice el Gemini), el caso es que hay que sacar todo en vectores y transponer uno de ellos para que Python se vea obligado (de forma super eficiente) a comparar elemento por elemento un vector con todo el contenido (uno por uno) del otro (contador de cosas misteriosas pero eficientes en python +1)

   temp=dfei["& I+ p"].values #en estos misteriosamente ya esta impuesto el numpy array y sale en una fila todo
   temp2=dfei["& I- p"].values
   tempref=df.index.to_numpy().reshape(-1,1) #es mejor imponer el array de numpy y hacerlo con el reshape para que no se queje el pandas de los truqitos, reshape (-1,1) lo hace vector columna
   
   #Localiza los tramos de estados de arriba (1)
   ranges=((temp<tempref) & (temp2>tempref)) #Pegrilo los puntos simpre mayor o menor nunca igual, ese punto pertenece justo al cambio, si no el kneeLocator no lo va a encontrar si es justo el extremo
   
       
   ranges=ranges.any(axis=1).astype(int) #con el any busca en todas las columnas los true que haya y los mete todos en un vector juntos (colapsa la matriz a un vector), el astype es para ponerlo en binario que se entiende mejor 
   df["State"]=ranges #Se mandan las etiquetas de las ramas del memristor al df original para tenerlas a mano, estado de potenciación 0 o depresión 1 


   print( "\033[1;35m" + str(len(dfe["Ewe+ p"])) + " Set and "+ str(len(dfe["Ewe- p"])) + " Lower/Upper loop state points labeled in " + str(df["nº ciclo"].nunique()) + " cycles" + "\033[0m\n")
   
 #Detección matemática de puntos SET/RESET
 #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   

   resultados = []

   for n in df["nº ciclo"].unique():
       
       #Para cuando el ajuste falle 
       fila_ciclo = {
           "nº ciclo": n,
           "Ewe Set": "-", "I Set": "-", "J Set": "-",
           "Ewe Reset": "-", "I Reset": "-", "J Reset": "-"
       }
       
       for modo in ["SET", "RESET"]:
           
           if modo == "SET":
               #Filtro 1er cuadrante (Ida positiva)
               rango = df[
                   (df["Ewe"] > 10e-5) & 
                   (df["Ewe"] <= dfe["Ewe+ p"][n]) & 
                   (df["nº ciclo"] == n) & 
                   (df["State"] == 0)
               ].copy(deep=False) 
           else:
               #Filtro 4º cuadrante (Ida negativa)
               rango = df[
                   (df["Ewe"] < -10e-5) & 
                   (df["Ewe"] >= dfe["Ewe- p"][n]) & 
                   (df["nº ciclo"] == n) & 
                   (df["State"] == 0) 
               ].copy(deep=False)
               rango = rango.iloc[::-1] #Inversión para origen en 0
               
           if len(rango) < 10:
               print("! The range contain less than 10 data points")#no es necesario pero por si algo se corrompe mejor tenerlo 
               continue 
               
           V_orig = rango["Ewe"].values#Para pasarlo a array de numpy
           I_orig = rango["I"].values
           J_orig = rango["J"].values  
           
           V_abs = np.abs(V_orig)
           I_abs = np.abs(I_orig)
           
           #Filtro anti-ruido (Savgol_filter standard CV)
           window = min(11, len(I_abs) - (len(I_abs) % 2 == 0))
           if window > 3:
               I_suave = savgol_filter(I_abs, window_length=window, polyorder=2) 
           else:
               I_suave = I_abs
               
           if np.max(I_suave) < thswitch:#Si no se llega a este umbral de corriente, se skipea la busqueda del punto set/rest del ciclo 
               print("! Cycle " + str(n) + " (" + modo + ") does not reach the current threshold")
               continue #es para la skip de la iteracion en el for mas cercano 
               
           idx_final = None #Lo ponemos para inciar la variable 
           
           #Búsqueda matemática del codo
           kneedle = KneeLocator(V_abs, I_suave,S=sens, curve="convex", direction="increasing", online=True)#El convex es porque el kneedle va al revés, la curva real es cóncava, misterios de la programación, el online es para usar menos memoria RAM
           codo_matematico = kneedle.knee 
           
           if codo_matematico is not None:
               #Soft switching
               idx_codo = np.where(V_abs == codo_matematico)[0][0] #Localiza el índice del codo matemático, [0][0] es para que saque el valor del array dentro de la tupla que devuelve where 
               corriente_codo = I_suave[idx_codo] #Valor de la corriente en el codo matemático
               corriente_base = np.min(I_suave[:idx_codo]) #Valor de la corriente en la base (antes del codo)
               
               umbral_despegue = corriente_base +thlitoff * (corriente_codo - corriente_base)
               
               idx_inicio_real = idx_codo
               for i in range(idx_codo, 0, -1): #Va para atras desde el codo para ver donde se despega respecto el umbral
                   if I_suave[i] <= umbral_despegue:
                       idx_inicio_real = i + 1
                       break
               idx_final = idx_inicio_real
               
           else: 
               #Hard switching
               saltos = np.diff(I_suave)#Diferencia entre valores consecutivos de la corriente (para detectar saltos)
               idx_hard = np.argmax(saltos) #Índice del mayor salto
               if saltos[idx_hard] > (np.max(I_suave) * thhard):#Si el mayor salto es mayor que el umbral
                   idx_final = idx_hard
                   
           #Extracción de coordenadas reales
           if idx_final is not None:
               if modo == "SET":
                   fila_ciclo["Ewe Set"] = V_orig[idx_final]
                   fila_ciclo["I Set"] = I_orig[idx_final]
                   fila_ciclo["J Set"] = J_orig[idx_final]
               else:
                   fila_ciclo["Ewe Reset"] = V_orig[idx_final]
                   fila_ciclo["I Reset"] = I_orig[idx_final]
                   fila_ciclo["J Reset"] = J_orig[idx_final]

       resultados.append(fila_ciclo)

   dfs = pd.DataFrame(resultados)

   dfs["Ewe+"] = dfe["Ewe+"] #Para ver los errores en el Ewe del potencioestato 
   dfs["Ewe-"] = dfe["Ewe-"]
   
 #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
   
 #Misc evolucion y estadística

   if describe== True:
       stats=dfs.describe()  
   else:
       stats=0
       pass
   
   
   if evolplot==True: #Graficador para ver como cambia el set y reset con los ciclos
       plt.plot(dfs["Ewe Set"],label="V set")
       plt.plot(abs(dfs["Ewe Reset"]),label="|V Reset|")
       
       plt.title("V conmutación vs nº de ciclo")
       plt.xlabel("nº ciclo")
       plt.ylabel(r"$V_c$ [V]")
       plt.legend(loc="upper right")
       plt.grid()
       
       plt.figure()# Nueva figura para pintar ahora la corriente vs nº ciclo 
       plt.plot(dfs["I Set"],label="I set")
       plt.plot(abs(dfs["I Reset"]),label="|I Reset|")
       
       plt.title("I conmutación vs nº de ciclo")
       plt.xlabel("nº ciclo")
       plt.ylabel(r"$I_c$ [mA]")
       plt.legend(loc="center right")
       plt.grid()

       plt.figure() #Ahora una de la resistencia on y off por ciclo
       plt.plot(dfs["Ewe Set"]/(dfs["I Set"]*1000),label="R set") 
       plt.plot(dfs["Ewe Reset"]/(dfs["I Reset"]*1000),label="R reset")    
       plt.title("R vs nº de ciclo")
       plt.xlabel("nº ciclo")
       plt.ylabel("R [Ω]")
       plt.legend(loc="upper right")
       plt.grid()       
   else:
       pass
   
   print( "\033[1;35m" + str(len(dfs["Ewe Set"])) + " Set and "+ str(len(dfs["Ewe Reset"])) + " Reset points labeled in " + str(df["nº ciclo"].nunique()) + " cycles" + "\033[0m\n")

   return(dfs,stats,dfe)   

def aproxcond(df,a,b,state,eq,cycle="Full",norm=False): # //Infra  
    """ 
    osz98N
    
    Esta función aplica la ecuación de condución y su aproximación para realizar el fit de los datos. Se ha separado de cond fit para poder realizar ajustes modulares, es un función infraestructura  
    
    Usando el estado 1/0, el modo Ohm y el tramo completo hasta set y reset es posible usar esto como un extractor de estados para exportar los datos y ajustar en otro lado. 
    
    df: Dataframe de pandas que contiene los datos a ajustar
    a:Voltaje inicio del tramo, indicar aqui el menor valor del rango
    b:Voltaje final del tramo, indicar aqui el mayor valor del rango
    state: Rama en la que hacer el ajuste, siendo 0 la de abajo (reset) y 1 la de arriba (set)
    cycle: Por defecto a full, si se indica otro valor solo usará ese ciclo (útil para revisar)
    
    Usando el estado 1/0, el modo Ohm y el tramo completo hasta set y reset es posible usar esto como un extractor de estados para exportar los datos y ajustar en otro lado. 
    
    """ 
    
    #Para poder tener batch processing y ajuste único sin volvernos locos filtramos dataframe cuando se usa el modo unico (para revisión)
    if cycle!="Full":
        df=df[df["nº ciclo"]==cycle]
    else:
        pass
    
    #Ahora localizamos el tramo accediendo por voltaje para un estado concreto
    df2=(df[["Ewe","J","I","nº ciclo"]].loc[(df["Ewe"]>=a) & (df["Ewe"]<=b) & (df["State"]==state)]).copy(deep=False) #deep=false crea df2 que contiene punteros a los datos pero no crea una copia física de estos 
    
    if norm==True:
        varnorm=["Ewe","I","J"] #Lista de cosas a normalizar
        df2[varnorm]=RobustScaler().fit_transform(df2[varnorm])#Noramlización para evitar que la diferentes magnitudes enmascaren dependencias (m muy pequeña que seria casi 0)
        #print(str(varnorm)+ " normalized !!")
    else:
        varnorm=0
    #Ecuaciones de fenómenos de conducción en aislantes, [J.M. Albella and J.M. Martinez. Física de Dieléctricos]
    #Atención se hará el valor absoluto de los valores, es decir el cuarto cuadrante se ve como si fuera le primero
    #--------------------------------------------------------------------------------------------------------------------------------------------------
    func_nl = None # Por defecto no hay función no lineal a ajustar
    if eq=="Poole-F":#Poole-Frenkel 
        df2["Poole-F"]=np.log10(abs(df2["J"]/df2["Ewe"]))
        df2["sqrt(V)"]=pow(abs(df2["Ewe"]),1/2)#1/2 random diffusion, 3/2 hopping and 3/4 diffusion transport 
             
        #Para generalizar los ajustes y graficos se hacen estas varibles que flagean los ejes X e Y
        x,y="sqrt(V)","Poole-F"
        xl,yl=r"$\sqrt{|V|}$ [$\sqrt{V}$]", r"$\log(|J/V|)$ [$\log(mA/(cm^2 \cdot V))$]"
        
    elif eq=="Schottky":#Schottky
        df2["Schottky"]=np.log10(abs(df2["J"]))
        df2["sqrt(V)"]=pow(abs(df2["Ewe"]),1/2)
        
        #Para generalizar los ajustes y graficos se hacen estas varibles que flagean los ejes X e Y
        x,y="sqrt(V)","Schottky"
        xl,yl=r"$\sqrt{|V|}$ [$\sqrt{V}$]", r"$\log(|J|)$ [$\log(mA/cm^2)$]"
        
    elif eq=="Ohm":#Ohmico R=V/I=dx/dy=m, ajuste por definición (V=IR) no se requieren calculos adicionales, tambien puede ser iónico 
        
        if norm!=True: #Si esa normalizado la afirmación es mentira
            print(r"Watch out! |m|=Resistance; This fit could be also for ionical as J ~ V")
        else:
            pass
        #Para generalizar los ajustes y graficos se hacen estas varibles que flagean los ejes X e Y
        x,y="Ewe","I"
        xl,yl="V [V]", "I [mA]"
        
    elif eq=="SCLC":#Limitación carga espacial 
        df2["V^2"]=pow(df2["Ewe"],2)
        df2["abs(J)"]=abs(df2["J"])
        
        #Para generalizar los ajustes y graficos se hacen estas varibles que flagean los ejes X e Y
        x,y="V^2","abs(J)"
        xl,yl=r"$V^2$ [$V^2$]", r"$|J|$ [$mA/cm^2$]"
    
    elif eq=="Sinh":# Tunel a bajo V aprox de la ec de Simons cuando el potencial es todavia cuadrado (F-N es con barrera triangular alto V)
        #df2["sinh(V)"]=np.sinh(df2["Ewe"])
        func_nl = lambda v, a, b: b * np.sinh(a * v) # Ecuación de tunel sinh() para meterla en scipy porque el statsmodels solo ajusta ec lineales
        
        x,y="Ewe","J"    
        xl,yl="V [V]", "J [mA/cm²]"
    
    elif eq=="F-N":#Fowler-Nordheim también conocido como emisión de campo, emisión de cátodo frío o tunel Nordheim-Fowler, si E<10^8 V/m es despreciable (para estos datos E~10^4 V/m) 
        #Si hay F-N estas ec aproximan el valor de W [eV] con la pendiente, es decir la altura de la barrera (sin lowerear) medida desde la interfase
        df2["1/V"]=abs(1/df2["Ewe"])
        df2["J/V^2"]=np.log(abs(df2["J"])/df2["Ewe"]**2)
        try:
            Wi=(df2.groupby("nº ciclo", group_keys=False)).apply(lambda dtemp: np.polyfit(dtemp["1/V"],dtemp["J/V^2"],1,cov=False), include_groups=False)# funcion de trabajo de la barrera o altura de la barrera en eV 
            W=Wi.mean()[0] #Solo queremos la pendiente promedio, la ordenada en el origen no sirve 
            if W<=0:
                raise ValueError("W<0")
            else:
                pass
            #Ecuación de Fowler-Nordheim aproximada 
            betta=1 # 1/nm pero es solo un factor, Voltage to barrier field, se ha puesto uno arbitrariamente que es como decir que todo el voltaje se usa para afectar la barrera
            B=6.83*betta*pow(W,3/2) #b es la cte segunda de Fowler Nordheim 
            df2["Campo"]=(df2["Ewe"]**2)*np.e**(-B/abs(df2["Ewe"])) 
            
            df2["abs(J)"]=abs(df2["J"])
            
            #Para generalizar los ajustes y graficos se hacen estas varibles que flagean los ejes X e Y
            x,y="Campo","abs(J)"
            xl,yl=r"$V^2 \cdot e^{-B/V}$ [$V^2$]", r"$|J|$ [$mA/cm^2$]"
            
        except: 
            print("Error: Barrier height don´t follow Fowler-Nordheim equation")
            x,y=0,0
            xl,yl="",""
    else:
        xl,yl="",""
        pass
    
    return(df2,x,y,xl,yl,varnorm,func_nl)

def linreg(df,a,b,state,eql=["Poole-F","Schottky","Ohm","SCLC","Sinh","F-N"],cycle="Full",norm=False,papermode=False,linear_tunnel=True):
    """ 
    osz98N
    
    Esta función realiza una regresión lineal simple empleando las ecuaiones de condución en aislantes  
     
    df: Dataframe de pandas que contiene los datos a ajustar
    a:Voltaje inicio del tramo, indicar aqui el menor valor del rango
    b:Voltaje final del tramo, indicar aqui el mayor valor del rango
    state: Rama en la que hacer el ajuste, siendo 0 la de abajo (reset) y 1 la de arriba (set)
    eq: Ecuación de conducicción a la que ajustar
    cycle: Por defecto a full, si se indica otro valor solo ajustará ese ciclo (útil para revisar)
    papermode: Si es True, solo grafica y ajusta el primer ciclo y uno de cada 10 para evitar saturación visual en figuras.
    
    """ 
    print("Linreg V1.0 osz98N \n" + "\033[1;33m Semi-Auto\033[0m"+ "\033[1;34m" +"\033[0m") 
    
    warnings_list = []
    
    # Configuración de la ventana con subplots
    num_eqs = len(eql)
    cols = 2 if num_eqs > 1 else 1
    rows = int(np.ceil(num_eqs / cols))
    fig_sub, axes_sub = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    fig_sub.canvas.manager.set_window_title('Ajustes de Conducción')
    if num_eqs > 1:
        axes_flat = axes_sub.flatten()
    else:
        axes_flat = [axes_sub]
        
    ax_list = [] # Para guardar las gráficas individualmente al final
    
    i=0
    for eq in eql:#Asi admite ajustes con multiples ecuaciones a la vez (batch fitting)
          
        #--------------------------------------------------------------------------------------------------------------------------------------------------
        df2,x,y,xl,yl,varnorm,func_nl=aproxcond(df,a,b,state,eq,cycle,norm)#Se instancia a las eq de condución, para tener las cuentas
        #--------------------------------------------------------------------------------------------------------------------------------------------------
        
        #Para figuras con menos ciclos y mas limpias de visualizar, decide qué ciclos graficar
        ciclos_plot = df["nº ciclo"].unique()[::25] if papermode else df["nº ciclo"].unique()

        if i==0 and norm==True:
            print("\033[1;31m Normalized mode on "+str(varnorm) +" variables!!"+"\033[0m"+"\n")
        else:
            pass
        
        #Ahora el ajuste y graficado, que son una parte comun por lo que se han generalizado
        
        try:#Por si el ajuste falla que no reviente todo 
           
           if eq == "Sinh" and linear_tunnel:
               xl_plot = "sinh(a·V)"
           else:
               xl_plot = xl

           ax = axes_flat[i]
           plt.sca(ax)
           
           resultados=[] #lista vacia para meter listas con todos los parametros de ajustes en cada ciclo 

           if func_nl is not None:#Funcion NO lineal, no funciona el statsmodels, utiliza la libreria scipy-----------------------------------------------------------------------------------------------------
               
               msg_nl = "\033[1;33mWarning: In " + eq + ", m=alpha and n=beta, non-linear fit \033[0m"
               if msg_nl not in warnings_list:
                   warnings_list.append(msg_nl)#Warning para no olvidarlo 
                   
               if eq == "Sinh":
                   msg_ion = "\033[1;33mWarning: In Sinh, la conducción iónica a campos E notables también se ajusta a sinh(), por lo que podría ser de tipo iónico.\033[0m"
                   if msg_ion not in warnings_list:
                       warnings_list.append(msg_ion)
               
               for c in df2["nº ciclo"].unique(): #Asi es como lo tenia en fitteador original que hice con scipy, en este caso es mejor un bucle por que hay muchas cosas que calcular y la lambda solo admite un calculo con una variable 
                   df2c = df2[df2["nº ciclo"] == c]
                   
                   if eq == "Sinh" and linear_tunnel:
                       x_pred_plot = np.sinh(df2c[x]) # Valor por defecto si el ajuste falla
                   else:
                       x_pred_plot = df2c[x]
                       
                   try:

                       #ftol, xtol y gtol reducen la tolerancia para evitar que se quede pillado en minimos locales, maxfev=numero máximo de iteraciones, bounds=limites para que no saque cosas raras, p0=punto inicial con el que empieza a buscar
                       popt, pcov = curve_fit(func_nl, df2c[x], df2c[y], p0=[1e-3, 1e-3], maxfev=10000, bounds=([0, 0], [np.inf, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
                       y_pred = func_nl(df2c[x], *popt)#El plot del ajuste no lineal, primero calcula la curva resultado y luego la grafica

                       if eq == "Sinh" and linear_tunnel:
                           x_pred_plot = np.sinh(popt[0] * df2c[x]) # Ajuste con el valor a

                       if c in ciclos_plot:
                           line, = plt.plot(x_pred_plot, y_pred, linestyle="--", linewidth=1.5, label=f"Ciclo {c}") 
                           plt.scatter(x_pred_plot, df2c[y], s=2, color=line.get_color(), alpha=0.7)

                       
                       #Aqui abajo calcula todos los parametros de ajuste ya que el scipy no lo hace (por eso migré los lineales a statsmodels)
                       ss_res = np.sum((df2c[y] - y_pred)**2)
                       ss_tot = np.sum((df2c[y] - np.mean(df2c[y]))**2)
                       r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                       errs = np.sqrt(np.diag(pcov))
                       
                       resultados.append([c, popt[0], popt[1], r_sq, errs[0], errs[1], np.nan, np.nan, np.sqrt(ss_res/max(1, len(df2c[y])-2))])#Añade los parametros en una lista a la lista 
                   except Exception as e:
                       print(f"Fit failed for cycle {c}: {e}")
                       if c in ciclos_plot:
                           plt.scatter(x_pred_plot, df2c[y], s=2, alpha=0.3, label=f"Ciclo {c} (Fallido)")

           else: #Funcion lineal, funciona el statsmodels-----------------------------------------------------------------------------------------------------------------------------------------------------

               ajustes=df2.groupby("nº ciclo").apply( lambda df2:( smf.ols( f"Q('{y}') ~ Q('{x}')",data=df2) ).fit(), include_groups=False) #f" para poder ponder las variables con la flag {} en un string tipo Pasty; Q('') para que vea eso como un nombre entero y vea palabras de operaciones
               #Con el apply se puede tratar el resultado de groupby como si fuera el df fisico ordenado por mismas filas (aunque es un objeto), la lambda hace de pipeline para que no meta todo de golpe en el ajuste 
               
               for c in df2["nº ciclo"].unique(): 
                   df2c=df2[df2["nº ciclo"]==c] 
                   if c in ajustes.index:
                       if c in ciclos_plot:
                           line, = plt.plot(df2c[x], ajustes[c].predict(df2c[x]), linestyle="--", linewidth=1.5, label=f"Ciclo {c}")
                           plt.scatter(df2c[x], df2c[y], s=2, color=line.get_color(), alpha=0.7)
                       
                       resultado=[c, ajustes[c].params[f"Q('{x}')"],ajustes[c].params["Intercept"],ajustes[c].rsquared,ajustes[c].bse[f"Q('{x}')"],ajustes[c].bse["Intercept"],ajustes[c].fvalue,ajustes[c].pvalues[f"Q('{x}')"],ajustes[c].mse_resid ** 0.5]#No olvidar el mse_resid**0.5 para que haga el error estandar de los residuos
                       resultados.append(resultado) #Lo de arriba llama a los parametros de interés y los mete en la lista para luego promediarlos como df
                   else:
                       if c in ciclos_plot:
                           plt.scatter(df2c[x], df2c[y], s=2, alpha=0.3, label=f"Ciclo {c} (Fallido)")
            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
            
           #El resto de cosas comunes 
           rama_str = " [d]" if state == 1 else " [r]"
           titulo_grafica = "Ajuste de " + str(df2["nº ciclo"].nunique()) + " ciclos a " + eq + rama_str
           
           plt.title(titulo_grafica)
           plt.xlabel(xl_plot)
           plt.ylabel(yl)
          
           resultados_df=pd.DataFrame(resultados,columns=["Ciclo", f"m ({x})","n","R^2","Err(m)","Err(n)","F", f"P ({x})","Err Std."])#En esta lista estan la mayoria de parametros importantes que da el modelo, abajo se promedian para evaluar todos los ciclos
           
           os.makedirs("TFM/resultados/logs/", exist_ok=True)
           rama_name = "set" if state == 1 else "reset"
           resultados_df.to_csv(f"TFM/resultados/logs/log_{eq}_{rama_name}.csv", index=False)
           
           raw_m = resultados_df[f"m ({x})"].mean()
           raw_n = resultados_df["n"].mean()
           raw_r2 = resultados_df["R^2"].mean()

           #---------------------------------------------------------------------------------------------------------
           #Cajita con el ajuste y los resultados en la grafica 
           N = len(resultados_df)
           std_m_stat = resultados_df[f"m ({x})"].std() / np.sqrt(N) if N > 1 else 0.0 #Aplica la formula de error estandar estadistico para los errores en n y m
           std_n_stat = resultados_df["n"].std() / np.sqrt(N) if N > 1 else 0.0
           std_r2 = resultados_df["R^2"].std(ddof=1) if N > 1 else 0.0 # Desviación estándar muestral (insesgada)

           err_fit_m = resultados_df["Err(m)"].mean()
           err_fit_n = resultados_df["Err(n)"].mean()
           
           err_fit_m = 0.0 if pd.isna(err_fit_m) else err_fit_m
           err_fit_n = 0.0 if pd.isna(err_fit_n) else err_fit_n
           
           std_m = np.sqrt(std_m_stat**2 + err_fit_m**2)
           std_n = np.sqrt(std_n_stat**2 + err_fit_n**2)

           def format_err(v, e): #Una función que se ha sacado la IA para que siempre haya una cifra significativa en los errores 
               if e == 0 or pd.isna(e):
                   return f"{v:.3e} ± 0.0"
               import math
               oom_e = math.floor(math.log10(abs(e)))#Se extrae el orden de magnitud del error y lo redondea a la baja 
               e_r = round(e, -oom_e) #Redondea el error a la baja con el orden de magnitud
               if e_r >= 10**(oom_e + 1): oom_e += 1 #Si el error se pasa al siguiente orden de magnitud se lo sumamos
               v_r = round(v, -oom_e)#Redondea el valor con el mismo orden que el error
               
               oom_v = math.floor(math.log10(abs(v_r))) if v_r != 0 else oom_e#Se extrae el orden de magnitud y lo redondea a la baja pero del valor ahora
               
               if oom_v != 0:
                   v_norm = v_r / 10**oom_v#Divide por el orden de magnitud para ajustar decimales según el orden de magnitud en el que esta (se queda con el numero ajustado a ese orden)
                   e_norm = e_r / 10**oom_v
                   dec = max(0, oom_v - oom_e)#Decide cuantos decimales poner en base a la diferencia entre el orden de magnitud del valor y el del error
                   return f"({v_norm:.{dec}f} ± {e_norm:.{dec}f}) $\\times 10^{{{oom_v}}}$"
               else:
                   dec = max(0, -oom_e)
                   return f"{v_r:.{dec}f} ± {e_r:.{dec}f}"

           m_str = format_err(raw_m, std_m)
           n_str = format_err(raw_n, std_n)

           if eq == "Sinh":
               texto_caja = f"J = b·sinh(a·V)\n$\\bar{{a}}$ = {m_str}\n$\\bar{{b}}$ = {n_str}\n$\\overline{{R^2}}$ = {raw_r2:.3f}\n$s(R^2)$ = {std_r2:.3f}"
           else:
               texto_caja = f"$\\bar{{m}}$ = {m_str}\n$\\bar{{n}}$ = {n_str}\n$\\overline{{R^2}}$ = {raw_r2:.3f}\n$s(R^2)$ = {std_r2:.3f}"
               
           if papermode:
               ax.legend(loc='best')
           else:
               # Motor dinámico de posicionamiento basado en Grid-Search exhaustivo
               try:
                   # Recopilar todos los puntos graficados (scatter y lines)
                   xdata, ydata = [], []
                   for c in ax.collections:
                       offsets = c.get_offsets()
                       if len(offsets) > 0:
                           xdata.extend(offsets[:, 0])
                           ydata.extend(offsets[:, 1])
                   for line in ax.lines:
                       lx = line.get_xdata()
                       ly = line.get_ydata()
                       if len(lx) > 1:
                           # Densificar las líneas: interpolar 50 puntos entre cada par de vértices
                           # Así el algoritmo de colisión "verá" un trazo continuo de píxeles y no solo los vértices sueltos
                           lx_dense = np.linspace(lx[:-1], lx[1:], 50).flatten()
                           ly_dense = np.linspace(ly[:-1], ly[1:], 50).flatten()
                           xdata.extend(lx_dense)
                           ydata.extend(ly_dense)
                       else:
                           xdata.extend(lx)
                           ydata.extend(ly)
                       
                   if not xdata:
                       raise ValueError
                       
                   xdata = np.array(xdata)
                   ydata = np.array(ydata)
                   
                   # Transformar los puntos de datos brutos a coordenadas visuales relativas [0, 1]
                   # Al usar las transformaciones nativas de Matplotlib, garantizamos que si la gráfica 
                   # tiene escalas logarítmicas (típico en I-V) o ejes invertidos, el mapa visual (hit grid)
                   # refleje EXACTAMENTE la posición donde el ojo humano ve las líneas dibujadas.
                   points = np.column_stack((xdata, ydata))
                   points_norm = ax.transAxes.inverted().transform(ax.transData.transform(points))
                   
                   x_norm = np.clip(points_norm[:, 0], 0, 0.999)
                   y_norm = np.clip(points_norm[:, 1], 0, 0.999)
                   
                   # Crear un mapa visual (Hit Grid) de 150x150 píxeles para medir el ÁREA de colisión real.
                   # Esto evita que una zona con miles de puntos agrupados asuste al algoritmo.
                   grid_size = 150
                   hit_grid = np.zeros((grid_size, grid_size), dtype=bool)
                   x_idx = np.int32(x_norm * grid_size)
                   y_idx = np.int32(y_norm * grid_size)
                   hit_grid[y_idx, x_idx] = True
                   
                   # Dejar que Matplotlib coloque la leyenda nativamente en el mejor lugar posible
                   leg = ax.legend(loc='best')
                   fig_sub.canvas.draw()
                   leg_bbox = leg.get_window_extent().transformed(ax.transAxes.inverted())
                   
                   # Para una tangencia perfecta sin solapes ni huecos grandes, calculamos el tamaño EXACTO
                   # de la caja de texto dibujándola temporalmente y obteniendo sus dimensiones reales.
                   dummy_text = plt.gca().text(0.5, 0.5, texto_caja, transform=plt.gca().transAxes, fontsize=10,
                                               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
                   fig_sub.canvas.draw()
                   txt_bbox = dummy_text.get_window_extent().transformed(ax.transAxes.inverted())
                   tw = txt_bbox.width
                   th = txt_bbox.height
                   
                   # La caja visual (bbox) siempre es más grande que el propio texto por el padding.
                   # Guardamos la distancia exacta entre el texto (0.5, 0.5) y las esquinas reales de la burbuja.
                   pad_left = 0.5 - txt_bbox.x0
                   pad_bottom = 0.5 - txt_bbox.y0
                   
                   dummy_text.remove()
                   
                   margin = 0.01
                   
                   best_score = float('inf')
                   best_pos = (0.020, 0.992, 'left', 'top') # default fallback
                   
                   # Escanear una malla de muy alta resolución (30x30 = 900 posiciones)
                   # Margen MUY ASIMÉTRICO para ajustes estéticos perfectos: 
                   # - Izquierda y Abajo (0.020): Margen amplio (2%) para no pisar las escalas.
                   # - Derecha (0.998): Margen casi nulo (0.2%) para pegarse y esquivar curvas.
                   # - Arriba (0.992): Margen ligero (0.8%) para que se vea la línea negra superior.
                   for bx in np.linspace(0.020, 0.998 - tw, 30):
                       for by in np.linspace(0.020 + th, 0.992, 30):
                           
                           # 1. Comprobar si choca con la leyenda (tangencia estricta usando el tamaño exacto)
                           overlap_x = not (bx + tw < leg_bbox.x0 or bx > leg_bbox.x1)
                           overlap_y = not (by - th > leg_bbox.y1 or by < leg_bbox.y0)
                           if overlap_x and overlap_y:
                               continue 
                               
                           # 2. Contar ÁREA visual que choca (píxeles de la grid) en lugar de puntos crudos
                           ix0 = int(max(bx - margin, 0) * grid_size)
                           ix1 = int(min(bx + tw + margin, 0.999) * grid_size)
                           iy0 = int(max(by - th - margin, 0) * grid_size)
                           iy1 = int(min(by + margin, 0.999) * grid_size)
                           
                           puntos_dentro = np.sum(hit_grid[iy0:iy1, ix0:ix1])
                           
                           # 3. Penalización por alejarse de los bordes (para que se pegue a las esquinas si puede)
                           dist_borde = min(bx, 1 - (bx + tw)) + min(by - th, 1 - by)
                           
                           # Puntuación: Lo más importante es que no pise líneas (x1000). El desempate es la cercanía al borde.
                           score = puntos_dentro * 1000 + dist_borde
                           
                           if score < best_score:
                               best_score = score
                               
                               # Ya no usamos anclaje dinámico porque el tamaño y los bordes están perfectamente controlados.
                               # Calculamos exactamente dónde tiene que anclarse el texto para que la "burbuja" visual
                               # (bbox) caiga de forma EXACTA y PÍXEL POR PÍXEL sobre las coordenadas [bx, by-th] que evaluamos.
                               final_ha = 'left'
                               final_va = 'bottom'
                               
                               final_x = bx + pad_left
                               final_y = (by - th) + pad_bottom
                               
                               best_pos = (final_x, final_y, final_ha, final_va)
                   
                   box_x, box_y, ha, va = best_pos
               except Exception:
                   # Fallback de emergencia si algo falla
                   box_x, box_y, ha, va = 0.020, 0.992, 'left', 'top'
                   
               plt.gca().text(box_x, box_y, texto_caja, transform=plt.gca().transAxes, fontsize=10,
                              verticalalignment=va, horizontalalignment=ha, 
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
                          
           ax_list.append((ax, titulo_grafica))
           #---------------------------------------------------------------------------------------------------------

           def format_val_err_table(v, e): 
               if pd.isna(e) or e == 0 or pd.isna(v):
                   return f"{v:.3e}" if not pd.isna(v) else "NaN", f"{e:.3e}" if not pd.isna(e) else "NaN"
               import math
               try:
                   oom_e = math.floor(math.log10(abs(e)))
                   e_r = round(e, -oom_e)
                   if e_r >= 10**(oom_e + 1): oom_e += 1
                   v_r = round(v, -oom_e)
                   
                   def to_str(val, oom_round):
                       if val == 0:
                           dec = max(0, -oom_round)
                           return f"{0.0:.{dec}f}"
                       oom = math.floor(math.log10(abs(val)))
                       if oom <= -3 or oom >= 4:
                           val_norm = val / 10**oom
                           dec = max(0, oom - oom_round)
                           return f"{val_norm:.{dec}f}e{oom:+03d}"
                       else:
                           dec = max(0, -oom_round)
                           return f"{val:.{dec}f}"

                   return to_str(v_r, oom_e), to_str(e_r, oom_e)
               except Exception:
                   return str(v), str(e)
           
           med_m_str, err_m_str = format_val_err_table(raw_m, resultados_df["Err(m)"].mean())
           med_n_str, err_n_str = format_val_err_table(raw_n, resultados_df["Err(n)"].mean())
           
           std_m_val = resultados_df[f"m ({x})"].std()
           std_n_val = resultados_df["n"].std()
           std_r2_val = resultados_df["R^2"].std()
           
           std_m_str = f"{std_m_val:.2e}" if not pd.isna(std_m_val) else "NaN"
           std_n_str = f"{std_n_val:.2e}" if not pd.isna(std_n_val) else "NaN"
           std_r2_str = f"{std_r2_val:.2e}" if not pd.isna(std_r2_val) else "NaN"
           
           med_r2_str = f"{raw_r2:.3f}" if not pd.isna(raw_r2) else "NaN"

           resultados=((pd.DataFrame([med_m_str, std_m_str, med_n_str, std_n_str, err_m_str, err_n_str, med_r2_str, std_r2_str])).T).set_axis(["med(m)","std(m)","med(n)","std(n)","Err(m)","Err(n)","med(R^2)","std(R^2)"],axis=1) #asi sale todo mas limpio, entendible y con promediado a todos los ciclos
           resultados.index=[eq] #El índice son los nombres de las eq
          
           print("\033[1;35m"+"Fitted to "+eq+"\033[0m\n")
           
           
       
        except Exception as mesg: #Lo de as mesg lo he dejado por si tengo otro bug 
            
            print("\033[1;35m"+"The data range can not be fitted to " + eq + "\033[0m\n"+ str(mesg))
            
            # Dibujar un cuadro indicando que no hubo convergencia en la gráfica que queda vacía
            rama_str = " [d]" if state == 1 else " [r]"
            titulo_grafica = "Ajuste de " + str(df2["nº ciclo"].nunique()) + " ciclos a " + eq + rama_str
            
            plt.sca(axes_flat[i])
            plt.title(titulo_grafica)
            plt.xlabel(xl)
            plt.ylabel(yl)
            
            texto_fallo = "Sin convergencia"
            plt.gca().text(0.5, 0.5, texto_fallo, transform=plt.gca().transAxes, fontsize=12,
                           verticalalignment='center', horizontalalignment='center', color='darkred',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='mistyrose', alpha=0.8, edgecolor='darkred'))
                           
            ax_list.append((axes_flat[i], titulo_grafica))
            
            # Para que si el ajuste falla que al concatenar no se vuelva loco Pandas 
            resultados=((pd.DataFrame(["NaN","NaN","NaN","NaN","NaN","NaN","NaN","NaN"])).T).set_axis(["med(m)","std(m)","med(n)","std(n)","Err(m)","Err(n)","med(R^2)","std(R^2)"],axis=1)# Para que si el ajuste falla que al concatenar no se vuelva loco Pandas 
            resultados.index=[eq]
            
            
        if i==0: #Si es la primera ecuacion bypassea la variable temporal y si no lo es concatena la temporal a la total
            resultadototal=resultados 
        else:
            resultadototal=pd.concat([resultadototal,resultados],axis=0)
        i=i+1 
    
    # Ocultar subplots vacíos si los hay
    for j in range(num_eqs, len(axes_flat)):
        axes_flat[j].set_visible(False)
        
    fig_sub.tight_layout(pad=1.0, w_pad=2.0, h_pad=2.0)
    fig_sub.canvas.draw()
    
    # Guardado automático individual de cada gráfica
    os.makedirs("TFM/resultados/auto/", exist_ok=True)
    for ax, titulo in ax_list:
        extent = ax.get_tightbbox(fig_sub.canvas.get_renderer()).transformed(fig_sub.dpi_scale_trans.inverted())
        titulo_valido = titulo.replace(":", "-").replace("/", "-")
        fig_sub.savefig("TFM/resultados/auto/" + titulo_valido + ".png", bbox_inches=extent.expanded(1.02, 1.02), dpi=600)
        
    plt.show(block=False)
    
    print(resultadototal)
    print("\n")
    for w in warnings_list:
        print(w)
        
    return(resultadototal)
    
def multilinreg(df,a,b,state,eqs,cycle="Full"):
    """ 
    osz98N
    
    Esta función realiza una regresión lineal múltiple empleando las ecuaiones de condución en aislantes  
     
    df: Dataframe de pandas que contiene los datos a ajustar
    a:Voltaje inicio del tramo, indicar aqui el menor valor del rango
    b:Voltaje final del tramo, indicar aqui el mayor valor del rango
    state: Rama en la que hacer el ajuste, siendo 0 la de abajo (reset) y 1 la de arriba (set)
    eqs: Ecuaciones con las que realizar regresión lineal méltiple
    cycle: Por defecto a full, si se indica otro valor solo ajustará ese ciclo (útil para revisar)
    
    """ 
    print("MultiLinreg V1.0 osz98N \n" + "\033[1;33m Manual\033[0m"+ "\033[1;34m" +"\033[0m") 
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------
    i=0
    x,y=(np.zeros(len(eqs))).astype("str"),(np.zeros(len(eqs))).astype("str")
    for eq in eqs:
        if i==0:
            df2,x[i],y[i],_,_,varnorm,func_nl=aproxcond(df,a,b,state,eq,cycle)#Se instancia a las eq de condución, para tener las cuentas
        else:
            df2temp,x[i],y[i],_,_,varnorm,func_nl=aproxcond(df,a,b,state,eq,cycle)
            colsfil=df2temp.columns.difference(df2.columns) #Se llena (o se puede llenar) el df de cosas duplicadas (que algunas son punteros), asi que solo metemos las que no estan que luego el pandas no sabe cual coger
            df2=pd.concat([df2,df2temp[colsfil]],axis=1)
        i=i+1

    #--------------------------------------------------------------------------------------------------------------------------------------------------
    #Ahora el ajuste y graficado, que son una parte comun por lo que se han generalizado
    
    #np.linalg.lstsq es peor solo da los coeficientes y ya, mejor usar statsmodels.api da mas cosas y es mas comodo no hay que calcular y hacer apaños como con polyfit <--- Descartado
    #df2["Poole-F"]=np.log10(abs((df2["Ewe"]/0.2)/df2["Ewe"])) #Una de las variables de regresion presenta colinealidad con la regresora!!!!
    #----------------------------------------------------------
    try:  
        
       #plt.scatter(df2[x],df2[y],s=0.5) #scatter que si no se pone loco a pintar lineas fantasma entre ciclos (no levanta el cursor de dibujo)   
       
       # 1. Limpiamos V2 de la influencia de V1
       modelo_limpieza = smf.ols("Schottky ~ Q('Poole-F')",df2).fit()
       df2["Schottky_puro"] = modelo_limpieza.resid
       
       #df2["log(Ewe)"]=np.log10(abs(df2["Ewe"]))
       #modelo_limpieza2 = smf.ols("Q('Poole-F') ~ Q('log(Ewe)')",df2).fit()
       #df2["Poole-F_puro"] = modelo_limpieza2.resid
       
       resultado=smf.mixedlm("Q('sqrt(V)') ~ Q('Schottky_puro') + Q('Poole-F') ",df2,groups=df2["nº ciclo"]).fit() #Para que lo tome como literal usar Q('') #,re_formula="~Q('Poole-F_puro')"
       cosa=resultado.params
       print(resultado.summary()) 
       
       # 1. Varianza de los efectos fijos (lo que explica el Tiempo)
       # Calculamos la varianza de las predicciones usando solo la parte fija
       result=resultado
       fixed_predict = result.model.predict(result.params)
       var_fixed = np.var(fixed_predict)

       # 2. Varianza de los efectos aleatorios (las diferencias entre Cerdos)
       # Statsmodels guarda esto en la matriz de covarianza de efectos aleatorios
       var_random = float(result.cov_re.iloc[0, 0])

       # 3. Varianza Residual (el ruido que sobra, lo que el modelo no sabe explicar)
       var_resid = result.scale

       # --- CÁLCULO DE LOS R CUADRADO ---
       total_var = var_fixed + var_random + var_resid

       # R2 Marginal (Solo el impacto del Tiempo)
       r2_marginal = var_fixed / total_var

       # R2 Condicional (El impacto del Tiempo + el agrupamiento por Cerdos)
       r2_condicional = (var_fixed + var_random) / total_var

       print(f"R^2 Marginal:    {r2_marginal:.4f}")
       print(f"R^2 Condicional: {r2_condicional:.4f}")
       
       """
       for i in range (df2["nº ciclo"].min(),(df2["nº ciclo"].max())+1):#cuidao con el +1 que como el range es hasta el max ciclo ese último no le incluye
       
           X=df2[y][df2["nº ciclo"]==i] 
           X=sm.add_constant(X) #Para que calcule la ordenada en el origen
           Y=df2[x[0]][df2["nº ciclo"]==i]
           
           resultado=sm.OLS(Y,X).fit() #sm.OLS no vale son var colineales
           parametros=resultado.summary()
           print(parametros)      
           print(i)
       """
       #plt.title("Ajuste de " + str(df["nº ciclo"].nunique()) + " ciclos a " + eq)
       #plt.xlabel(str(x))
       #plt.ylabel(str(y))
       #plt.show()
       
       
       print("\033[1;35m"+"Fitted to "+str(eqs)+"\033[0m\n")
       
    except Exception as msg:
        
        print("\033[1;35m"+"The data range can not be fitted to " + str(eqs)+print (msg)+"\033[0m\n")

    return(cosa)    


#----------------------------------------Analisis de los datos------------------------------------------------------------------------------------- 
#-----------------------------------------------------------------------------------------------------------------------
#df1=leer_enJ("TFM/Solo 1 poroso/Set canónico/C-SPEIS_J80_T1s_SPEIS_01.mpt","SPEIS",desde=1,hasta=4) #del set canónico 


df1=leer_enJ("TFM/Solo 1 poroso/Un poroso para gobernalos a todos/MP_J60_T120s_01.mpt","CV")
df1s,info,df1e=labelmem(df1,evolplot=False)
#print(info)
#JV(df1,"Representación J-V",log=False)
#df1["R"]=df1["Ewe"]/(df1["I"]/1000)  

ajuste=linreg(df1,1,3,0,papermode=True) 
#ajuste=linreg(df1,3,4.9,1,papermode=True) 
#ajuste=linreg(df1,-7,-1,0,papermode=True)

"""
#ajuste2=multilinreg(df1,3,4.8,0,["Poole-F","Schottky"])
df1["R"]=df1["Ewe"]/(df1["I"]/1000)
r1=(df1[["R"]].loc[(df1["Ewe"]>=0.5) & (df1["Ewe"]<=4.9) & (df1["State"]==0)])
r2=(df1[["R"]].loc[(df1["Ewe"]>=0.5) & (df1["Ewe"]<=4.9) & (df1["State"]==1)])
df1["Dm"]=r1.reset_index(drop=True)-r2.reset_index(drop=True)
maxDM=df1[["Dm","nº ciclo"]].groupby(["nº ciclo"]).max()
print(maxDM)
plt.plot(maxDM.index,maxDM["Dm"])
plt.show()
"""
plt.show()
####################################################################################################################################################
"""
spice = pd.read_csv("TFM/Solo 1 poroso/Un poroso para gobernalos a todos/memristor poroso mejor ajuste de I al DC sweep de medida.txt", sep="\t")

spice["Ewe"] = spice["V(vi)"]
spice["J"] = (spice["Ix(U1:TE)"] / (np.pi*(0.2/2)**2))*1000
spice["nº ciclo"] = 1
spice["State"] = 0

# Inventamos una etiqueta de estado basada en la derivada del voltaje
diff_ewe = np.diff(spice["Ewe"])
diff_ewe = np.append(diff_ewe, diff_ewe[-1]) # Igualamos longitud para la mascara
creciente = diff_ewe > 0
decreciente = diff_ewe <= 0

#JV(spice, "Curva J-V SPICE", nciclo=1, log=False)
#JV(df1,"Curva J-V  J45 ; t120s",log=False, new_fig=False)
fig, ax = plt.subplots()
ax.scatter(df1["Ewe"],df1["J"], s=2, color="gray", edgecolors="none", zorder=1, alpha=1, label="Experimental")
ax.scatter(spice["Ewe"][creciente], spice["J"][creciente], s=0.5, color="tab:blue", edgecolors="none", zorder=2, alpha=1, label="Simulado (Creciente)")
ax.scatter(spice["Ewe"][decreciente], spice["J"][decreciente], s=0.5, color="tab:orange", edgecolors="none", zorder=3, alpha=1, label="Simulado (Decreciente)")
ax.set_title("Histéresis experimetal vs simulada")
ax.set_xlabel("V [V]")
ax.set_ylabel("J [mA/cm²]")
ax.grid(True)
leg = ax.legend(markerscale=5)
for line in leg.get_lines():
    line.set_linewidth(4.0) # Hace las líneas de la leyenda más gruesas para que se vean bien

# Inset (mini gráfica) al estilo paper
import matplotlib.patches as mpatches
# Recuadro exterior blanco para aislar el inset y sus textos de la cuadrícula principal
rect = mpatches.Rectangle((0.08, 0.22), 0.50, 0.54, transform=ax.transAxes, 
                         facecolor='white', edgecolor='black', linewidth=1.3, zorder=2.5)
ax.add_patch(rect)

# [left, bottom, width, height] en fracciones del eje principal
axins = ax.inset_axes([0.18, 0.32, 0.35, 0.35])
axins.set_zorder(3) # Por encima del recuadro
axins.scatter(spice["Ewe"][creciente], (spice["V(n003)"]*1000)[creciente], s=0.5 ,color="tab:blue", edgecolors="none", zorder=3)
axins.scatter(spice["Ewe"][decreciente], (spice["V(n003)"]*1000)[decreciente], s=0.5, color="tab:orange", edgecolors="none", zorder=4)
axins.set_title(r"V vs x", fontsize=9)
axins.set_xlabel("V [V]", fontsize=8)
axins.set_ylabel(r"x", fontsize=8)
axins.grid(True, alpha=0.5)
axins.tick_params(labelsize=8)
axins.set_facecolor('none') # Fondo transparente para que se vea el blanco del recuadro

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

spice2 = pd.read_csv("TFM/Solo 1 poroso/Un poroso para gobernalos a todos/memristor poroso mejor ajuste de I DC sweep de 430mV por s.txt", sep="\t")
spice3 = pd.read_csv("TFM/Solo 1 poroso/Un poroso para gobernalos a todos/memristor poroso mejor ajuste de I DC sweep de 700mV por s.txt", sep="\t")

spice2["Ewe"] = spice2["V(vi)"]
spice2["J"] = (spice2["Ix(U1:TE)"] / (np.pi*(0.2/2)**2))*1000

spice3["Ewe"] = spice3["V(vi)"]
spice3["J"] = (spice3["Ix(U1:TE)"] / (np.pi*(0.2/2)**2))*1000

# Subfigura a) 430 mV/s
axs[0, 0].scatter(spice2["Ewe"], spice2["J"], s=0.5, color="tab:orange", edgecolors="none", label="Simulación a 430 mV/s")
axs[0, 0].set_title("a)", loc="left", fontsize=16, fontweight="bold")
axs[0, 0].set_xlabel("V [V]", fontsize=14)
axs[0, 0].set_ylabel("J [mA/cm²]", fontsize=14)
axs[0, 0].tick_params(labelsize=12)
axs[0, 0].grid(True)
leg_a = axs[0, 0].legend(markerscale=10, loc="upper left", fontsize=14)
for handle in leg_a.legend_handles: handle.set_alpha(1.0)

# Subfigura b) 500 mV/s (del dataframe spice cargado arriba)
axs[0, 1].scatter(spice["Ewe"], spice["J"], s=0.5, color="tab:green", edgecolors="none", label="Simulación a 500 mV/s")
axs[0, 1].set_title("b)", loc="left", fontsize=16, fontweight="bold")
axs[0, 1].set_xlabel("V [V]", fontsize=14)
axs[0, 1].set_ylabel("J [mA/cm²]", fontsize=14)
axs[0, 1].tick_params(labelsize=12)
axs[0, 1].grid(True)
leg_b = axs[0, 1].legend(markerscale=10, loc="upper left", fontsize=14)
for handle in leg_b.legend_handles: handle.set_alpha(1.0)

# Subfigura c) 700 mV/s
axs[1, 0].scatter(spice3["Ewe"], spice3["J"], s=0.5, color="tab:blue", edgecolors="none", label="Simulación a 700 mV/s")
axs[1, 0].set_title("c)", loc="left", fontsize=16, fontweight="bold")
axs[1, 0].set_xlabel("V [V]", fontsize=14)
axs[1, 0].set_ylabel("J [mA/cm²]", fontsize=14)
axs[1, 0].tick_params(labelsize=12)
axs[1, 0].grid(True)
leg_c = axs[1, 0].legend(markerscale=10, loc="upper left", fontsize=14)
for handle in leg_c.legend_handles: handle.set_alpha(1.0)

# Subfigura d) Vacía para la foto
axs[1, 1].axis('off')
axs[1, 1].set_title("d) Circuito simulado", loc="left", fontsize=16, fontweight="bold")

plt.tight_layout()
plt.show()
"""
"""
#SPEIS Zfit revisar que esta con EXCEL
#dfS=leer_enJ("TFM/Solo 1 poroso/Un poroso para gobernalos a todos/SPEIS 1mes despues/J45_t120s_SPEIS_todojunto_01.mpt","Z fit SPEIS")

# Para leer excel que no es lo habitual (con cabecera dinámica)
df_temp = pd.read_excel("TFM/Solo 1 poroso/Un poroso para gobernalos a todos/SPEIS 1mes despues/datos oscar.xlsx", header=None)
idx = df_temp[df_temp.eq('V').any(axis=1)].index[0]
dfS = pd.read_excel("TFM/Solo 1 poroso/Un poroso para gobernalos a todos/SPEIS 1mes despues/datos oscar.xlsx", header=idx)
dfS = dfS.rename(columns={"V": "Ewe", "a2": "a3", "Q2": "Q3"})
diff_ewe = np.diff(abs(dfS["Ewe"]))
diff_ewe = np.append(diff_ewe, diff_ewe[-1]) # Igualamos longitud para la mascara
creciente = diff_ewe >= 0
decreciente = diff_ewe < 0


# Mantener la escala proporcional al voltaje barrido (y agrupar lecturas consecutivas idénticas)
x_val = abs(dfS["Ewe"].diff().fillna(0)).cumsum()

is_integer = dfS['Ewe'] % 1 == 0
unique_integer_x = x_val[is_integer].drop_duplicates()

ticks_idx = unique_integer_x.values
tick_labels = [f"{int(dfS['Ewe'].loc[i])}" for i in unique_integer_x.index]

fig_a, ax1 = plt.subplots()

ax1.scatter(x_val[creciente], dfS["a3"][creciente], s=15, color="tab:blue", edgecolors="none", label="a3 Creciente")
ax1.plot(x_val.where(creciente), dfS["a3"].where(creciente), color="tab:blue", linestyle="-", linewidth=1.0)
ax1.scatter(x_val[decreciente], dfS["a3"][decreciente], s=15, color="tab:orange", edgecolors="none", label="a3 Decreciente")
ax1.plot(x_val.where(decreciente), dfS["a3"].where(decreciente), color="tab:orange", linestyle="-", linewidth=1.0)

ax1.legend(loc="upper center", ncol=2, markerscale=2)
ax1.set_title("Parámetro a del CPE vs V")
ax1.set_xlabel("V [V]")
ax1.set_ylabel("a")
ax1.set_xticks(ticks_idx)
ax1.set_xticklabels(tick_labels)

ymin1, ymax1 = ax1.get_ylim()
ax1.set_ylim(ymin1, ymax1 + (ymax1 - ymin1) * 0.25)
ax1.grid()

fig_q, ax2 = plt.subplots()

ax2.scatter(x_val[creciente], dfS["Q3"][creciente], color="tab:green", s=15, edgecolors="none", label="Q3 Creciente")
ax2.plot(x_val.where(creciente), dfS["Q3"].where(creciente), color="tab:green", linestyle="-", linewidth=1.0)
ax2.scatter(x_val[decreciente], dfS["Q3"][decreciente], color="tab:red", s=15, edgecolors="none", label="Q3 Decreciente")
ax2.plot(x_val.where(decreciente), dfS["Q3"].where(decreciente), color="tab:red", linestyle="-", linewidth=1.0)

ax2.legend(loc="upper center", ncol=2, markerscale=2)
ax2.set_title("Parámetro Q del CPE vs V")
ax2.set_xlabel("V [V]")
ax2.set_ylabel("Q [F$\cdot$s$^{a-1}$]")
ax2.set_xticks(ticks_idx)
ax2.set_xticklabels(tick_labels)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

ymin2, ymax2 = ax2.get_ylim()
ax2.set_ylim(ymin2, ymax2 + (ymax2 - ymin2) * 0.25)
ax2.grid()

fig2, ax3 = plt.subplots()
ax4 = ax3.twinx()

# Cargar datos experimentales adicionales
df_extra = pd.read_excel("TFM/Solo 1 poroso/Un poroso para gobernalos a todos/SPEIS 1mes despues/mas cosas para Oscar.xlsx")
df_extra["J"] = df_extra["I"] / 0.2

# Determinar las ramas de forma robusta frente al ruido experimental usando los picos globales
idx_max = df_extra["V"].idxmax()
idx_min = df_extra["V"].idxmin()
macro_dV = pd.Series(1, index=df_extra.index)
macro_dV.loc[idx_max:idx_min] = -1

# Creciente = alejarse de 0 (V y dV tienen el mismo signo)
creciente_full = (df_extra["V"] * macro_dV) >= 0

# Dividir el potencial entre dos para que el rango de 10 sea de 5
df_extra["V"] = df_extra["V"] / 2
creciente_extra = creciente_full.copy()
decreciente_extra = ~creciente_extra

# Resetear índices para graficar sin problemas
df_extra.reset_index(drop=True, inplace=True)
creciente_extra.reset_index(drop=True, inplace=True)
decreciente_extra.reset_index(drop=True, inplace=True)

ax3.scatter(df_extra["V"][creciente_extra], df_extra["J"][creciente_extra], s=4, color="tab:blue", edgecolors="none", label="J Creciente")
ax3.plot(df_extra["V"].where(creciente_extra), df_extra["J"].where(creciente_extra), color="tab:blue", linestyle="-", linewidth=1.0)
ax3.scatter(df_extra["V"][decreciente_extra], df_extra["J"][decreciente_extra], s=4, color="tab:orange", edgecolors="none", label="J Decreciente")
ax3.plot(df_extra["V"].where(decreciente_extra), df_extra["J"].where(decreciente_extra), color="tab:orange", linestyle="-", linewidth=1.0)

ax4.scatter(dfS["Ewe"][creciente], dfS["R2"][creciente], color="tab:green", marker="x", label="R2 Creciente")
ax4.plot(dfS["Ewe"].where(creciente), dfS["R2"].where(creciente), color="tab:green", linestyle="--", linewidth=1.0)
ax4.scatter(dfS["Ewe"][decreciente], dfS["R2"][decreciente], color="tab:red", marker="x", label="R2 Decreciente")
ax4.plot(dfS["Ewe"].where(decreciente), dfS["R2"].where(decreciente), color="tab:red", linestyle="--", linewidth=1.0)

lines_3, labels_3 = ax3.get_legend_handles_labels()
lines_4, labels_4 = ax4.get_legend_handles_labels()

import matplotlib.lines as mlines
# Crear handles personalizados para J para que el punto en la leyenda sea grande, pero sin afectar a las cruces de R2
custom_j_crec = mlines.Line2D([], [], color='tab:blue', marker='o', linestyle='-', markersize=6)
custom_j_decr = mlines.Line2D([], [], color='tab:orange', marker='o', linestyle='-', markersize=6)
lines_3 = [custom_j_crec, custom_j_decr]

lines2 = [item for sublist in zip(lines_3, lines_4) for item in sublist]
labels2 = [item for sublist in zip(labels_3, labels_4) for item in sublist]
# Desplazamos un poco a la izquierda con bbox_to_anchor y ampliamos el hueco entre columnas con columnspacing
ax3.legend(lines2, labels2, loc="upper center", bbox_to_anchor=(0.45, 1.0), ncol=2, columnspacing=4.0, markerscale=1)

plt.title("J y R2 vs V")
ax3.set_xlabel("Ewe [V]")
ax3.set_ylabel("J [mA/cm²]")
ax4.set_ylabel("R2 [$\Omega$]")
ax3.set_xlim(-5, 5)

# Ampliar el límite superior de ambos ejes un 30% para que la leyenda no pise los datos
ymin3, ymax3 = ax3.get_ylim()
ax3.set_ylim(ymin3, ymax3 + (ymax3 - ymin3) * 0.3)
ymin4, ymax4 = ax4.get_ylim()
ax4.set_ylim(ymin4, ymax4 + (ymax4 - ymin4) * 0.3)

ax3.grid()
plt.show()
"""
#-----------------------------------------------------------------------------------------------------------------------
"""
# Gráfica de Nyquist (SPEIS)
df_speis = leer_enJ("TFM/Solo 1 poroso/Un poroso para gobernalos a todos/SPEIS 1mes despues/J45_t120_SPEIS_01.mpt", "SPEIS", desde=2)

try:
    fig_nyq, (ax_nyq, ax_zoom) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'wspace': 0.40})
    
    # Separamos los ciclos detectando saltos bruscos en la frecuencia de barrido
    rango_f = df_speis["Freq"].max() - df_speis["Freq"].min()
    ciclos = (df_speis["Freq"].diff().abs() > rango_f * 0.5).cumsum()

    # Se grafica Z_re vs Z_-im (Nyquist standard) iterando por ciclos para "levantar el lápiz"
    num_ciclos = len(ciclos.unique())
    colores = plt.cm.viridis(np.linspace(0.85, 0, num_ciclos)) # 0.9 para evitar el amarillo muy claro sobre el fondo blanco
    
    for i, c in enumerate(ciclos.unique()):
        segmento = df_speis[ciclos == c]
        
        if i == 0:
            lbl = "5V"
        elif i == num_ciclos - 1 and num_ciclos > 1:
            lbl = "0V"
        else:
            lbl = None
            
        ax_nyq.plot(segmento["Z_re"], segmento["Z_-im"], linewidth=1.5, marker='o', markersize=3, color=colores[i], zorder=1, label=lbl)
        
    ax_nyq.set_title(r"Representación de Nyquist de 5V $\rightarrow$ 0V", fontsize=16)
    ax_nyq.set_xlabel("Re(Z) [$\Omega$]", fontsize=14)
    ax_nyq.set_ylabel("-Im(Z) [$\Omega$]", fontsize=14)

    #ax_nyq.grid(True)
    ax_nyq.legend(loc="upper left", fontsize=14) 
    ax_nyq.ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)
    ax_nyq.tick_params(labelsize=12)
    
    # Ajuste de fuente de notación científica
    ax_nyq.xaxis.get_offset_text().set_fontsize(12)
    ax_nyq.yaxis.get_offset_text().set_fontsize(12)
    
    ax_nyq.set_aspect('equal', adjustable='datalim') # Misma escala en X e Y para Nyquist
    ax_nyq.set_xlim(-0.5e5, 4e5) # Ajustado el espacio negativo y positivo del eje X

    # --- COORDENADAS A DEFINIR POR EL USUARIO ---
    center_x = 49500  # <--- PONER AQUI COORDENADA X
    center_y = 4100  # <--- PONER AQUI COORDENADA Y
    span_x = 5200   # <--- RANGO EN X DEL INSET
    span_y = 5000   # <--- RANGO EN Y DEL INSET

    for i, c in enumerate(ciclos.unique()):
        segmento = df_speis[ciclos == c]
        ax_zoom.plot(segmento["Z_re"], segmento["Z_-im"], linewidth=1.5, marker='o', markersize=3, color=colores[i], zorder=3)
    
    ax_zoom.set_xlim(center_x - (span_x/2), center_x + (span_x/2))
    ax_zoom.set_ylim(center_y - (span_y/2), center_y + (span_y/2))
    ax_zoom.set_title("Región de 1-20 Khz a 4V", fontsize=16)
    ax_zoom.set_xlabel("Re(Z) [$\Omega$]", fontsize=14)
    ax_zoom.set_ylabel("-Im(Z) [$\Omega$]", fontsize=14)
    ax_zoom.ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)
    ax_zoom.grid(True, alpha=0.5)
    ax_zoom.tick_params(labelsize=12)
    
    # Ajuste de fuente de notación científica
    ax_zoom.xaxis.get_offset_text().set_fontsize(12)
    ax_zoom.yaxis.get_offset_text().set_fontsize(12)
    
    # Cuadro de indicación en la gráfica original para saber de dónde viene el zoom
    rect_zoom, conectores = ax_nyq.indicate_inset_zoom(ax_zoom, edgecolor="black")
    for conector in conectores:
        conector.set_visible(False)
    
    # Para que no quede mucho espacio entre ambas
    fig_nyq.tight_layout()
    plt.show()
except NameError:
    print("\033[1;33mInfo: Para plotear Nyquist, asegúrate de tener una variable 'df_speis' con tus datos.\033[0m")
"""
#-----------------------------------------------------------------------------------------------------------------------

#Notas 
"""
P-F         n=1 0.4-2.8V        Batch 3-4.8V hay combinacion lineal de P-F y Scho 
Scho        n=1 1.9-5V          Batch 3-4.8V hay combinacion lineal de P-F y Scho

Scho                            4.8-3V domina este <-------
Ohm         n=1 hay 4.75-4.25   Batch no esta claro pero su m es bastante mas alto, quizas con T se puede mirar si hay algo ionico para que ajuste perfecto
P-F                             3-1V domina este casi seguro <-------

P-F                             Batch (-)1-3V pendiente negativa y mucha peor m (m=-0.3) aunque es el mejor R^2
Ohm                             Batch no esta claro pero su m es bastante mas alto, quizas con T se puede mirar si hay algo ionico para que ajuste perfecto
Scho                            Batch (-)3-4-6.9V domina Scho pero hay mucho de Pole y en el ultimo tramo Ohm tiene un R^2 muy bueno!

Scho                            Batch (-)3-6.9V domina Scho y hay poco de Pole? y casi nada de Ohm bien!; nada de Ohm = pendiente de 0 (ni sale) y R^2 malo 
P-F                             Batch (-)1-3V se ha tragado el P-F solo Scho, posiblemente el efecto diodo del poroso no le esta dejando conducir a Pole 
"""




