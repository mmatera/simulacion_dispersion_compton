from os import mkdir
from pathlib import Path
import sim
from sim import Experimento, Detector, ShieldCilindrico, BlancoCilindrico, Evento


from time import time
import numpy as np
import pickle


# parametros
sim.DEBUG = False
# Descomentar esta línea par forzar a que la fuente sólo
# emita a lo largo del eje z.
# sim.CONO_FUENTE = tuple((.95, 1.))
# Descomentar esta línea para no agregar el foton de 1200
# sim.HERALDO = False
# sim.EMITIR_EN_PLANO = True

DURACION = 240000000
RESET = True


cilindro1 = BlancoCilindrico(
        posicion=np.array([0, 0, 10.0]),
        alto=2,
        radio=1,
        lambda_absorcion=100000.0,
        lambda_dispersion=3.003,
    )

cilindro2 = BlancoCilindrico(
        posicion=np.array([0, 0, -10.0]),
        alto=2,
        radio=1,
        lambda_absorcion=100000.0,
        lambda_dispersion=3.003,
    )

shield1 = ShieldCilindrico(
        eje="z",
        lambda_absorcion=.01,  # La longitud de penetracion para el plomo ~.1cm
        lambda_dispersion=.01,  # La longitud de penetracion para el plomo ~.1cm
        radio_interior=1,
        radio_exterior=2,
        alto=10,
    )


start = Detector(eficiencia=1.0, posicion=np.array([ 10, 0,  10]), radio=1)
stop  = Detector(eficiencia=1.0, posicion=np.array([  0, 10, -10]), radio=1)
blancos = [shield1, cilindro1, cilindro2]  # [shield1, cilindro1, cilindro2]


if __name__ == "__main__":
    clean_start = RESET
    folder = f"results/{Path(__file__).name[:-3]}"
    print(folder)
    print("Cono fuente:", sim.CONO_FUENTE)
    print("Emitir en plano:", sim.EMITIR_EN_PLANO)
    print("Heraldo:", sim.HERALDO)
    
    try:
        mkdir(folder)
    except:
        pass

    if not clean_start:
        try:
            with open(f"{folder}/checkpoint.pkl", "rb") as f_in:
                exp = pickle.load(f_in)
                print("cuentas almacenadas:", sum(exp.coincidencias))
        except Exception:
            print(
                f"{folder}/checkpoint.pkl no encontrado. Generando un nuevo experimento."
            )
            clean_start = True

    if clean_start:
        exp = Experimento(
            step=0.05,
            flujo_eventos=10000.,  # 3x10^-7 cuentas/ns en el experimento.
            start=start,
            stop=stop,
            blancos=blancos,
            retardo_start=1,
            retardo_stop=10,
            folder=folder,
            checkpoint_interval=20.0,
        )

    t_start = time()
    exp.simular(DURACION)
    t_stop = time()
    exp.check_point(show_plots=True)
