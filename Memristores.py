# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 10:37:28 2026

@author: osz98N
"""
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

import statsmodels.formula.api as smf

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
        skipeables=56
        nombres=["Freq","Z_re","Z_-im","Z","Phase(Z)","Time","Ewe","I"]
        columnas_csv=[0,1,2,3,4,5,6,7]
    elif tecnica=="CV":
        skipeables=48 #la primera fila es justo donde arranca, tiene datos que no valen, skipeo de 48 (lo que pone en el ,mpt)
        nombres=["Ewe","I","nº ciclo"]
        columnas_csv=[7,8,9]
    elif tecnica=="Z fit SPEIS":
         skipeables=11
         nombres=["Q3","a3","nº ciclo","Ewe","I"]
         columnas_csv=[3,4,6,7,8]
    else:
        print("Error tipo no reconocido")
        
#conversión de todo a -1, asi es mas intuitivo de usar, si es -1 el while se ejecuta hasta que se rompa
    if hasta == "todo":
        hasta=-1
        
#se incluye un desde, asi se puede leer solo un archivo concreto    
    if desde<9:
        archivo=archivo.replace("_0"+str(1)+"_","_0"+str(desde)+"_",1)
       
    else:
        archivo=archivo.replace("_0"+str(1)+"_","_"+str(desde)+"_",1)

    i=desde #no es necesario pero me gusta mas el while con la i      
 #---------------------------------------------------------------------       
    while (i<hasta+1 or hasta== -1):#se lee todos los archivos flageados con ese nombre hasta donde se quiera      
#lectura del archivo mpt: !!separador decimal ",", las 56 primeras filas son basura, he overaideado los nombres!!
        
        try:#si ya no hay mas que se leer se sale del bucle
            dftemp=pd.read_csv(archivo, sep="\t",skiprows=skipeables,decimal=',',header=0,names=nombres,usecols=columnas_csv)
        except:#si no hay archivo a tomar por culo no lee mas 
            break
        
        try:#si es la primera lectura el try catch definira df con el temporal. 
            df=pd.concat([df,dftemp],axis=0)
        except NameError:
            df=dftemp
            
        #pasa al siguiente archivo numerado con ese nombre (siguiente ciclo de medidas)
        if i<9:
            archivo=archivo.replace("_0"+str(i)+"_","_0"+str(i+1)+"_",1)
            i=i+1 #cuidado que como se olvide se hace bucle infinito 
        else:
            archivo=archivo.replace("_0"+str(i)+"_","_"+str(i+1)+"_",1)
            i=i+1
#conversión  a J-----------------------------------------------------------------
    #D=0.2  #diametro del contacto de oro en cm = 0.2cm por defecto
    A=np.pi*(D/2)**2 #area del contacto de oro en cm^2
    #df["I"]=df["I"]/1000 #Se pasa a Amperios para tener todo en S.I.!!!!!
    df["J"]=df["I"]/A #J=I/A 
#Textito chulo 
    print( "\033[1;35m"+ str(df.shape) +" Data points has been loaded" + " on " +tecnica + " Mode" + "\033[0m\n")     
#---------------------------------------------------------------------------------------
    return (df)

def JV (df,titulo="Curva J-V",nciclo="todo",log=False):
   """
    osz98N
    
    Esta función hace la gráfica J-V de un dataframe 
    
    df: Dataframe de pandas donde estan los datos del potencioestato y la columna con la J (véase leer_enJ)
    titulo: Titulo de la gráfica (que será el nombre el archivo tambíen), por defecto "Curva J-V: 1poroso"
    Precaución cuando se trabaje con multiporosos si se usa el por defecto
    nciclo: número del ciclo a leer, por ahora solo puede ir ciclo a ciclo en los archivos medidos por CV
    
   """ 
  #Escala normal/logy--------------------------------------------------------------------------------
  
   if log==True: #Para poder tener escala normal y logaritmica de forma compacta 
        plt.yscale("log") #Escala log en y
        df["J"]=abs(df["J"])
   else:
        pass
  
   #Ciclos a graficar uno/todos--------------------------------------------------------------------------------
   
   if nciclo!="todo": #grafica solo un ciclo (solo CV)
   
       plt.plot(df["Ewe"][df["nº ciclo"]==nciclo],df["J"][df["nº ciclo"]==nciclo], linewidth=2, markersize=1) # SPEIS no tiene n ciclo!!!!!!!!!!!!!!
       
   else: #graficar todo  (para SPEIS solo funciona este) 
   
       plt.plot(df["Ewe"],df["J"], linewidth=1, markersize=1, color="gray")
       
       plt.plot(df["Ewe"][df["nº ciclo"]==df["nº ciclo"].min()],df["J"][df["nº ciclo"]==df["nº ciclo"].min()], linewidth=1, markersize=1)#se pone como en los papers el primer y ultimo ciclo remarcados, lo pinta dos veces (gris y en color) pero prefiero ahorrarme el if 
       plt.plot(df["Ewe"][df["nº ciclo"]==df["nº ciclo"].max()],df["J"][df["nº ciclo"]==df["nº ciclo"].max()], linewidth=1, markersize=1)
       
  
   #Info de la grafica y otros -------------------------------------------------------------------------------- 
   
   plt.title(titulo)
   plt.xlabel("Ewe [V]")
   plt.ylabel("J [mA/cm^2]")
   
   plt.grid()
   
   plt.savefig(dpi=1000,fname="TFM/resultados/"+titulo+".png")
   plt.show() 
   return()

def labelmem (df,mode="fast",describe=True,evolplot=False):
   """ 
    osz98N
    
    Esta función busca los puntos de Set y Reset del memristor por definición matemática (en set/reset hay un max/min local en corriente eléctrica I)
    Considere que si el memristor falla el ciclo de lectura escritura debe eliminar ese ciclo manualmente df.drop()
    
   df: Pandas dataframe a computar, obligatorio contar con columna etiquetadora de ciclos (solo modo CV por ahora)
   mode: fast/safe, solo usar fast en datos donde se tenga la seguridad de que el set/reset coincide con el punto de máxima corriente (memristor ideal)
   describe: Descrición estadística de los ciclos on/off
   """ 
   print("Labelmem V1.0 osz98N \n" + "\033[1;33m Full Auto\033[0m"+ "\033[1;34m    Mode:"+mode +"\033[0m") 
   
   if mode == "fast": #Aka el modo por máximos locales, muy sencillo y eficiente (el primero en implementarse) pero poco robusto si los memristores no son ideales (de libro)
       #La solucion mas sencilla y eficiente es tener una lista de máximos y mínimos de (Ewe Set, Reset) para cada ciclo agrupados con groupby (Ewe+,- es fijo en el potencioestato, pero tiene error)

       dfe=(pd.concat([df.groupby(by=["nº ciclo"]).max(),df.groupby(by=["nº ciclo"]).min()],axis=1, ignore_index=True)).set_axis(["Ewe+","I Set","J Set","Ewe-","I Reset","J Reset"],axis=1) #Pegrilo, puntos de set y reset computados por definición matemática (máximos y mínimos absolutos en el lazo)
       
       #Para tener arriba la matriz limpia, se hace una separada de indirecciones (realmente posicion en index de df1) de los valores de I que interesan

       dfei=(pd.concat([df.groupby(by=["nº ciclo"])["I"].idxmax(),df.groupby(by=["nº ciclo"])["I"].idxmin()], axis=1)).set_axis(["& I Set","& I Reset"],axis=1)

       #Ahora se busca los valores de Ewe set y reset, partiendo de los puntos de Imax (picos de corriente en el caso de hard switching), como tenemos las indirecciones esto es ahora mas facil (contador de horas buscando esta solucion porque no me acuerdo de los comandos 6h) ya que  hay valores repetidos entre ciclos!!

       dfe.insert(1 ,"Ewe Set",(((df[["Ewe","nº ciclo"]]).iloc[dfei["& I Set"]]).set_index("nº ciclo")).set_axis(["Ewe Set"],axis=1))

       dfe.insert(5 ,"Ewe Reset",(((df[["Ewe","nº ciclo"]]).iloc[dfei["& I Reset"]]).set_index("nº ciclo")).set_axis(["Ewe Reset"],axis=1))

       #Ahora se mira la estadistica de los ciclos para comprobar reproducibilidad
       if describe== True:
           stats=dfe.describe()  
       else:
           stats=0
           pass
       
   elif mode == "safe":
        print("En proceso, usar np.gradient")
       
       
   else:
       print("Modo no reconocido")
       
   #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   #Ahora se etiquetan las ramas de set y reset, lo que permite evitar tener todo junto cuando se ajuste en tramos de voltaje (que nativamente ya diferencia entre lazo positivo y negativo)
   
   #Es imposible hacer en Pandas una comparación (una col) valor por valor de un dataframe a otro (con diferente longitud) tambien valor por valor, es decir revisar si un valor pertenece a un rango de valores dados por dos columnas. (Es posible pero con una cosa que te puede reventar la memoria del PC) 
   #Esto en SQL se llama Non-equi Join (eso dice el Gemini), el caso es que hay que sacar todo en vectores y transponer uno de ellos para que Python se vea obligado (de forma super eficiente) a comparar elemento por elemento un vector con todo el contenido (uno por uno) del otro (contador de cosas misteriosas pero eficientes en python +1)


   temp=dfei["& I Set"].values #en estos misteriosamente ya esta impuesto el numpy array y sale en una fila todo
   temp2=dfei["& I Reset"].values
   #como no este todo en vectores al comparar revienta, por los tamaños diferentes se pone muy nervioso incluso cuando no es una comparacion fila a fila exacta
   tempref=df.index.to_numpy().reshape(-1,1) #es mejor imponer el array de numpy y hacerlo con el reshape para que no se queje el pandas de los truqitos, reshape (-1,1) lo hace vector columna
   ranges=((temp<=tempref) & (temp2>=tempref)) #Se coge el rango en el que se esta en el estado 1 (set), sale una matriz al comparar vector fila con vector columna (matriz Booleana), pegrilo matriz de [len(df.index)Xlen(puntos_set/reset)]
   ranges=ranges.any(axis=1).astype(int) #con el any busca en todas las columnas los true que haya y los mete todos en un vector juntos (colapsa la matriz a un vector), el astype es para ponerlo en binario que se entiende mejor

   df["State"]=ranges #Se mandan las etiquetas de las ramas al df original para tenerlas a mano
   
   if evolplot==True: #Graficador para ver como cambia el set y reset con los ciclos
       plt.plot(dfe["Ewe Set"],label="V set")
       plt.plot(abs(dfe["Ewe Reset"]),label="|V Reset|")
       
       plt.title("V disparo vs nº de ciclo")
       plt.xlabel("nº ciclo")
       plt.ylabel("V [V]")
       plt.legend(loc="center right")
       plt.grid()
       
       plt.figure()# Nueva figura para pintar ahora la corriente vs nº ciclo 
       plt.plot(dfe["I Set"],label="I set")
       plt.plot(abs(dfe["I Reset"]),label="|I Reset|")
       
       plt.title("I disparo vs nº de ciclo")
       plt.xlabel("nº ciclo")
       plt.ylabel("I [mA]")
       plt.legend(loc="center right")
       plt.grid()
       
   else:
       pass
   
   print( "\033[1;35m" + str(len(dfei["& I Set"])) + " Set and "+ str(len(dfei["& I Reset"])) + " Reset points labeled in " + str(df["nº ciclo"].nunique()) + " cycles" + "\033[0m\n")

   return(dfe,stats)   

def aproxcond(df,a,b,state,eq,cycle="Full",norm=True): # //Infra  
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
    if eq=="Pole-F":#Pole-Frenkel 
        df2["Pole-F"]=np.log10(abs(df2["J"]/df2["Ewe"]))
        df2["sqrt(V)"]=pow(abs(df2["Ewe"]),1/2)
             
        #Para generalizar los ajustes y graficos se hacen estas varibles que flagean los ejes X e Y
        x,y="sqrt(V)","Pole-F"
        
    elif eq=="Scho":#Schottky
        df2["Schottky"]=np.log10(abs(df2["J"]))
        df2["sqrt(V)"]=pow(abs(df2["Ewe"]),1/2)
        
        #Para generalizar los ajustes y graficos se hacen estas varibles que flagean los ejes X e Y
        x,y="sqrt(V)","Schottky"
        
    elif eq=="Ohm":#Ohmico R=V/I=dx/dy=m, ajuste por definición (V=IR) no se requieren calculos adicionales, tambien puede ser iónico 
        
        if norm!=True: #Si esa normalizado la afirmación es mentira
            print(r"Watch out! |m|=Resistance; This fit could be also for ionical as J ~ V")
        else:
            pass
        #Para generalizar los ajustes y graficos se hacen estas varibles que flagean los ejes X e Y
        x,y="Ewe","I"
        
    elif eq=="Chargelim":#Limitación carga espacial 
        df2["V^2"]=pow(df2["Ewe"],2)
        df2["abs(J)"]=abs(df2["J"])
        
        #Para generalizar los ajustes y graficos se hacen estas varibles que flagean los ejes X e Y
        x,y="V^2","abs(J)"
        
    elif eq=="F-N":#Fowler-Nordheim también conocido como emisión de campo, emisión de cátodo frío o tunel Nordheim-Fowler, si E<10^8 V/m es despreciable (para estos datos E~10^4 V/m) 
        #Si hay F-N estas ec aproximan el valor de W [eV] con la pendiente, es decir la altura de la barrera (sin lowerear) medida desde la interfase
        df2["1/V"]=abs(1/df2["Ewe"])
        df2["J/V^2"]=np.log(abs(df2["J"])/df2["Ewe"]**2)
        try:
            Wi=(df2.groupby("nº ciclo", group_keys=False)).apply(lambda dtemp: np.polyfit(dtemp["1/V"],dtemp["J/V^2"],1,cov=False))# funcion de trabajo de la barrera o altura de la barrera en eV 
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
            
        except: 
            print("Error: Barrier height don´t follow Fowler-Nordheim equation")
            x,y=0,0
    else:
        pass
    
    return(df2,x,y,varnorm)

def linreg(df,a,b,state,eql=["Pole-F","Scho","Ohm","Chargelim","F-N"],cycle="Full",norm=True):
    """ 
    osz98N
    
    Esta función realiza una regresión lineal simple empleando las ecuaiones de condución en aislantes  
     
    df: Dataframe de pandas que contiene los datos a ajustar
    a:Voltaje inicio del tramo, indicar aqui el menor valor del rango
    b:Voltaje final del tramo, indicar aqui el mayor valor del rango
    state: Rama en la que hacer el ajuste, siendo 0 la de abajo (reset) y 1 la de arriba (set)
    eq: Ecuación de conducicción a la que ajustar
    cycle: Por defecto a full, si se indica otro valor solo ajustará ese ciclo (útil para revisar)
    
    """ 
    print("Linreg V1.0 osz98N \n" + "\033[1;33m Semi-Auto\033[0m"+ "\033[1;34m" +"\033[0m") 
    
    i=0
    for eq in eql:#Asi admite ajustes con multiples ecuaciones a la vez (batch fitting)
          
        #--------------------------------------------------------------------------------------------------------------------------------------------------
        df2,x,y,varnorm=aproxcond(df,a,b,state,eq,cycle,norm)#Se instancia a las eq de condución, para tener las cuentas
        #--------------------------------------------------------------------------------------------------------------------------------------------------
        
        if i==0 and norm==True:
            print("\033[1;31m Normalized mode on "+str(varnorm) +" variables!!"+"\033[0m"+"\n")
        else:
            pass
        
        #Ahora el ajuste y graficado, que son una parte comun por lo que se han generalizado
        
        try:#Por si el ajuste falla que no reviente todo 
            
           plt.scatter(df2[x],df2[y],s=0.5) #scatter que si no se pone loco a pintar lineas fantasma entre ciclos (no levanta el cursor de dibujo)
           
           ajustes=df2.groupby("nº ciclo").apply( lambda df2:( smf.ols( f"Q('{y}') ~ Q('{x}')",data=df2) ).fit()) #f" para poder ponder las variables con la flag {} en un string tipo Pasty; Q('') para que vea eso como un nombre entero y vea palabras de operaciones
           #Con el apply se puede tratar el resultado de groupby como si fuera el df fisico ordenado por mismas filas (aunque es un objeto), la lambda hace de pipeline para que no meta todo de golpe en el ajuste 
           
           resultados=[] 
           for c in ajustes.index: #Por cada ajuste de la lista grafica lo que has predecido y saca los parametros de ajuste para promediarlos a todos los ciclos 
               df2c=df2[df2["nº ciclo"]==c] #Dice el inteligente artificial que si lo pongo con df2[x][df2["nº ciclo"]==c] el statsmodels se va a poner exquisito con que es una serie y no un df (explosion)
               plt.plot(df2c[x], ajustes[c].predict(df2c[x]))
               
               resultado=[ajustes[c].params[f"Q('{x}')"],ajustes[c].params["Intercept"],ajustes[c].rsquared,ajustes[c].bse[f"Q('{x}')"],ajustes[c].bse["Intercept"],ajustes[c].fvalue,ajustes[c].pvalues[f"Q('{x}')"],ajustes[c].mse_resid ** 0.5]#No olvidar el mse_resid**0.5 para que haga el error estandar de los residuos
               resultados.append(resultado) #Lo de arriba llama a los parametros de interés y los mete en la lista para luego promediarlos como df
                
           plt.title("Ajuste de " + str(df["nº ciclo"].nunique()) + " ciclos a " + eq)
           plt.xlabel(str(x))
           plt.ylabel(str(y))
           plt.show()
          
           resultados=pd.DataFrame(resultados,columns=[f"m ({x})","n","R^2","Err(m)","Err(n)","F", f"P ({x})","Err Std."])#En esta lista estan la mayoria de parametros importantes que da el modelo, aunque se pondran según hagan falta 
           
           resultados=((pd.DataFrame([resultados[f"m ({x})"].mean(),resultados[f"m ({x})"].std(),resultados["n"].mean(),resultados["n"].std(),resultados["Err(m)"].mean(),resultados["Err(n)"].mean(),resultados["R^2"].mean(),resultados["R^2"].std()])).T).set_axis(["med(m)","std(m)","med(n)","std(n)","Err(m)","Err(n)","med(R^2)","std(R^2)"],axis=1) #asi sale todo mas limpio, entendible y solo lo necesario
           resultados.index=[eq] #El índice son los nombres de las eq
           resultados=(resultados.apply(pd.to_numeric,errors="coerce").round(3))#Sale como objeto el df y si no haces esto no se redondea
          
           print("\033[1;35m"+"Fitted to "+eq+"\033[0m\n")
           
           
       
        except Exception as mesg: #Lo de as mesg lo he dejado por si tengo otro bug 
            
            print("\033[1;35m"+"The data range can not be fitted to " + eq + "\033[0m\n"+ str(mesg))
            
            # Para que si el ajuste falla que al concatenar no se vuelva loco Pandas 
            resultados=((pd.DataFrame([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])).T).set_axis(["med(m)","std(m)","med(n)","std(n)","Err(m)","Err(n)","med(R^2)","std(R^2)"],axis=1)# Para que si el ajuste falla que al concatenar no se vuelva loco Pandas 
            resultados.index=[eq]
            
            
        if i==0: #Si es la primera ecuacion bypassea la variable temporal y si no lo es concatena la temporal a la total
            resultadototal=resultados 
        else:
            resultadototal=pd.concat([resultadototal,resultados],axis=0)
        i=i+1 
    
    print(resultadototal)
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
            df2,x[i],y[i]=aproxcond(df,a,b,state,eq,cycle)#Se instancia a las eq de condución, para tener las cuentas
        else:
            df2temp,x[i],y[i]=aproxcond(df,a,b,state,eq,cycle)
            colsfil=df2temp.columns.difference(df2.columns) #Se llena (o se puede llenar) el df de cosas duplicadas (que algunas son punteros), asi que solo metemos las que no estan que luego el pandas no sabe cual coger
            df2=pd.concat([df2,df2temp[colsfil]],axis=1)
        i=i+1

    #--------------------------------------------------------------------------------------------------------------------------------------------------
    #Ahora el ajuste y graficado, que son una parte comun por lo que se han generalizado
    
    #np.linalg.lstsq es peor solo da los coeficientes y ya, mejor usar statsmodels.api da mas cosas y es mas comodo no hay que calcular y hacer apaños como con polyfit <--- Descartado
    #df2["Pole-F"]=np.log10(abs((df2["Ewe"]/0.2)/df2["Ewe"])) #Una de las variables de regresion presenta colinealidad con la regresora!!!!
    #----------------------------------------------------------
    try:  
        
       #plt.scatter(df2[x],df2[y],s=0.5) #scatter que si no se pone loco a pintar lineas fantasma entre ciclos (no levanta el cursor de dibujo)   
       
       # 1. Limpiamos V2 de la influencia de V1
       modelo_limpieza = smf.ols("Schottky ~ Q('Pole-F')",df2).fit()
       df2["Schottky_puro"] = modelo_limpieza.resid
       
       #df2["log(Ewe)"]=np.log10(abs(df2["Ewe"]))
       #modelo_limpieza2 = smf.ols("Q('Pole-F') ~ Q('log(Ewe)')",df2).fit()
       #df2["Pole-F_puro"] = modelo_limpieza2.resid
       
       resultado=smf.mixedlm("Q('sqrt(V)') ~ Q('Schottky_puro') + Q('Pole-F') ",df2,groups=df2["nº ciclo"]).fit() #Para que lo tome como literal usar Q('') #,re_formula="~Q('Pole-F_puro')"
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
#df1=leer_enJ("TFM/Solo 1 poroso/Set canónico/C-SPEIS_J80_T1s_01_SPEIS_01.mpt","SPEIS",desde=1,hasta=1) #del set canónico 
dfS=leer_enJ("TFM/Solo 1 poroso/Un poroso para gobernalos a todos/SPEIS 1mes despues/J45_t120s_SPEIS_tojunto_zfitparam_01.mpt","Z fit SPEIS",desde=1,hasta=1)

i=0
j=16
for m in range(1,5):
    plt.plot(dfS["Ewe"][i:j],dfS["a3"][i:j],label="a3 in cycle"+str(m))
    i=i+16
    j=j+16
#plt.plot(dfS["Ewe"],dfS["a3"])    
plt.legend()
plt.title("a vs V")
plt.xlabel("Ewe [V]")
plt.ylabel("a ")
plt.grid()





"""
df1=leer_enJ("TFM/Solo 1 poroso/Un poroso para gobernalos a todos/MP_J60_T120s_01.mpt","CV",desde=1,hasta=1)
JV(df1,"Curva J-V  1 poroso",3)
df1e,info=labelmem(df1,evolplot=False)


a=3
b=4.8
s=0
#ajuste=linreg(df1,a,b,s,norm=False) #La normalización quizas la tenia que haber aplicado despues cuando ya se tiene los datos pasados por las ec, algo raro hace cambiar el modelo de normaliza

#ajuste2=multilinreg(df1,3,4.8,0,["Pole-F","Scho"])
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


