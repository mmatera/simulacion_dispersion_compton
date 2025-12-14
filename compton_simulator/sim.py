from datetime import datetime
from time import time
import numpy as np
from numpy.random import default_rng
from matplotlib import pyplot as plt
import pickle
from typing import List
from numpy.linalg import norm

import numba

PI = 3.1415926

rng = default_rng()

# parametros
DEBUG = True
DURACION = 1000000
RESET = True

CONO_FUENTE = tuple((-0.99999999, 1))
EMITIR_EN_PLANO = False
HERALDO = True


@numba.njit
def evol_photons(positions, momenta, energias, dt):
    """
    Actualiza las posiciones, evolucionando un tiempo dt
    """
    positions = positions + dt * 29.9 * momenta / energias
    return positions
    
@numba.njit
def dentro_detector(pos_foton, x_detector, r_detector):
    rel_pos = (pos_foton - x_detector)
    distances = np.sum(rel_pos**2, axis=1)
    inside = distances < r_detector**2
    return [i for i, state in enumerate(inside) if state]


@numba.njit
def dentro_cilindro(pos_foton, x_cil, r_cil, h_cil):
    """
    Dada una lista de puntos, determina los índices correspondientes
    a puntos dentro de un cilindro centado en x_cil, de radio r_cil
    y alto h_cil
    """
    rel_pos = (pos_foton - x_cil)
    inside = np.abs(rel_pos[:, 2])<.5*h_cil
    in_radius = np.sum(rel_pos[inside,:2]**2, axis=1)< r_cil**2
    inside[inside] = np.logical_and(inside[inside], in_radius)
    return [i for i, state in enumerate(inside) if state]


@numba.njit
def dentro_anillo(pos_foton, x_cil, r_int_cil:float, r_ext_cil:float, h_cil:float, eje:int)->List[int]:
    """
    Dada una lista de puntos, determina los índices correspondientes
    a puntos dentro de un anillo cilíndrico centado en x_cil, de radio interior
    r_in_cil, radio exterior r_ext_cil y alto h_cil.
    """
    half_h_cil=.5*h_cil
    rad_in_sq = r_int_cil**2
    rad_ex_sq = r_ext_cil**2
    count = 0
    mask = np.empty(len(pos_foton), np.bool_)
    
    for i, pos in enumerate(pos_foton):
        rel_pos = pos-x_cil
        if eje==0:
            dz, dy, dx = rel_pos
        elif eje==1:
            dx, dz, dy = rel_pos
        else:
            dx, dy, dz = rel_pos

        if abs(dz)> half_h_cil:
            mask[i] = False
            continue
        rsq = dx**2+dy**2 
        if rsq > rad_ex_sq:
            mask[i]=False
            continue
        if rsq < rad_in_sq:
            mask[i]=False
            continue
        mask[i]=True
        count+=1

    return [i for i, val in enumerate(mask) if val]



def time_with_units(t) -> str:
    if t > 1.0e9:
        return f"{int(t * 1.e-6)/1000} s"
    if t > 1.0e6:
        return f"{int(t * 1.e-3)/1000} ms"
    if t > 1.0e3:
        return f"{int(t) * 1.e-3} us"
    if t < 1.0e-3:
        return f"{int(t * 1.**3)} ps"
    return f"{int(t*1000)/1000} ns"


def prob_detection(eficiencia, n):
    """
    Dada una eficiencia, y el cociente n=2 R/(c delta_t),
    determina la probabilidad de detección en el intervalo delta_t
    para un detector de radio $R$ si su eficiencia es `eficiencia`.
    """
    if eficiencia > 10.0:
        return 1.0
    # print_debug("Calculando la probabilidad de deteccion")

    def eficiencia_media(p: float):
        """
        Devuelve la eficiencia y su derivada en función de la
        probabilidad de detección.
        """
        # Si n es R/(c delta t),
        # y p es la probabilidad de
        # detección en cada paso delta t
        # calculamos la probabilidad de
        # detección media (eficiencia)  y su derivada como
        if p >= 1:
            return 1.0, 0.0
        if p == 0.0:
            return 0.0, 0.0
        q = 1 - p
        s = q**n
        log_s = np.log(s)
        e_media = 1 - 2 * (1 - s * (1 - log_s)) / log_s**2
        de_media = 4 * n / (q * log_s**3) * ((s - 1) + s * (0.5 * log_s - 1) * log_s)
        return e_media, de_media

    steps = 20
    y0 = eficiencia
    p = eficiencia
    while steps:
        y, dy = eficiencia_media(p)
        delta_y = y0 - y
        if abs(delta_y) < 0.01:
            break
        new_p = delta_y / dy + p
        if p > 1:
            p = 0.5 + 0.5 * p
        elif p < 0:
            p = 0.5 * p
        else:
            p = new_p
        steps = steps - 1
    return p


def print_debug(*args):
    if DEBUG:
        print(*args)


def KN_ThetaPhi(en=1):
    """
    Devuelve una orientación al azar,
    con una distribución proporcional
    a la sección eficaz diferencial de Klein Nishina.
    """
    cos_theta = KN_CosTheta(en)
    theta = np.arccos(cos_theta)
    PI = np.pi

    def dof_phi(phi, en):
        """
        Densidad de probabilidad para phi, para una 
        dada energía.
        """
        la = 1 / (1 + en * (1 - cos_theta))
        a = 1 - cos_theta**2 # Sin(theta)**2
        w = a / (la + 1.0 / la - a)
        return (1 + w * np.cos(2 * phi)) / (2.0 * PI)

    def cof_phi(phi, en):
        la = 1 / (1 + en * (1 - cos_theta))
        a = 1 - cos_theta**2
        w = a / (la + 1.0 / la - a)
        u = 2 * phi
        return 0.5 + 0.25 * (u + w * np.sin(u)) / PI

    y0 = rng.uniform()
    x0 = (2.0 * y0 - 1.0) * PI
    rem_steps = 20
    while rem_steps:
        y1 = cof_phi(x0, en)
        dy = y0 - y1
        dx = dy / dof_phi(x0, en)
        if abs(dx) < 0.01:
            # print_debug(f"  convergence achieved in {20-rem_steps}")
            break
        x1 = x0 + dx
        if x1 > PI:
            x0 = 0.5 * (PI + x0)
        elif x1 < -PI:
            x0 = 0.5 * (-PI + x0)
        else:
            x0 = x1
        rem_steps -= 1
    return theta, x0


def KN_CosTheta(en=1):
    """
    Produce un Cos(theta) distribuido según la distribución de
    Klein - Nishina para energía en * mc^2.
    """

    def dof_costheta(costheta, en=1):
        if en == 0:
            return 3 / 8 * (1 + costheta**2)
        else:
            la = 1 / (1 + en * (1 - costheta))
            val = en**3 * la**2 * (la + 1 / la - (1 - costheta**2))
            w = 1 + 2 * en
            norm = 2 * en * (2 + en * (1 + en) * (8 + en)) / (w**2)
            norm += np.log(w) * (en * (en - 2) - 2)
            return val / norm

    def cof_costheta(costheta, en=1):
        if en == 0:
            return (4 + 3 * costheta + costheta**3) / 8.0
        else:
            return (
                0.01
                * sum(dof_costheta(u, en) for u in np.linspace(-1, costheta, 100))
                * (1.0 + costheta)
            )

    y0 = rng.uniform()
    x0 = 2.0 * y0 - 1.0
    rem_steps = 20
    while rem_steps:
        y1 = cof_costheta(x0, en)
        dy = y0 - y1
        dx = dy / dof_costheta(x0, en)
        if abs(dx) < 0.01:
            # print_debug(f"  convergence achieved in {20-rem_steps}")
            break
        x1 = x0 + dx
        if x1 > 1:
            x0 = 0.5 + 0.5 * x0
        elif x1 < -1:
            x0 = 0.5 * x0 - 0.5
        else:
            x0 = x1
        rem_steps -= 1
    return x0


class Foton:
    def __init__(self, posicion, momento, polarizacion=None, singlete=False):
        self.historia = [posicion]
        self.posicion = posicion
        self.momento = momento
        self.polarizacion = polarizacion
        self.energia = norm(momento)
        self.singlete = singlete

    def __repr__(self):
        if self.polarizacion:
            return "foton: " + self.posicion.__repr__() + ", " + self.polarizacion
        elif self.singlete:
            return "foton: " + self.posicion.__repr__() + " entangled"
        return "foton: " + self.posicion.__repr__()

    def draw_setup_top(self, ax, color="yellow"):
        if self.energia == 0:
            return

        x, y, z = self.posicion
        px, py, pz = self.momento / self.energia

        if abs(z**2 - 100) > 50:
            return
        # Elige el color como función de la energía. Los fotones de baja
        # energía (los dispersados) viran al rojo. Los 511 son amarillos, y
        # los más energéticos se van haciendo amarillo cada vez más pálido.
        color = (1.0, min(1.0, self.energia / 511), max(0, 1 - 511 / self.energia))
        historia_x = [p[0] for p in self.historia] + [x]
        historia_y = [p[1] for p in self.historia] + [y]

        ax.scatter([x], [y], color=color)
        ax.plot(historia_x, historia_y, color=color)

    def draw_setup_2d(self, ax, color="yellow"):
        if self.energia == 0:
            return

        x, y, z = self.posicion
        # Sólo dibuja los fotones en un slice de 2 cm en
        # torno al plano y=0
        if abs(y) > 1:
            return
        px, py, pz = self.momento / self.energia
        assert abs(pz) < 2 and abs(px) < 2, [self.energia, self.momento]
        # Elige el color como función de la energía. Los fotones
        # de baja energía (los dispersados) viran al rojo. Los 511 son
        # amarillos, y los más energéticos se van haciendo amarillo
        # cada vez más pálido.
        color = (1.0, min(1.0, self.energia / 511), max(0, 1 - 511 / self.energia))

        ax.scatter([x], [z], color=color)
        historia_x = [p[0] for p in self.historia] + [x]
        historia_z = [p[2] for p in self.historia] + [z]

        ax.plot(historia_x, historia_z, color=color)

    def aniquilar(self):
        self.posicion = np.array([10000.0, 0.0, 0.0])
        self.momento = np.array([0.0, 0.0, 0.0])
        self.energia = 0

    def compton(self):
        self.historia.append(self.posicion)
        # print_debug("   Dispersion compton")
        p, pol = self.momento, self.polarizacion
        if pol is None:
            pol = "V" if rng.uniform() < 0.5 else "H"

        # print_debug("Polarizacion: ", pol)

        def sortear(en):
            """
            Produce un par theta, phi distribuidos de acuerdo con
            la KN differential cross section
            """
            return KN_ThetaPhi(en)

        def rotar(vector, eje, angulo):
            assert abs(1 - norm(eje)) < 0.001
            return (
                np.dot(vector, eje) * eje
                - np.cos(angulo) * np.cross(eje, np.cross(eje, vector))
                + np.sin(angulo) * np.cross(eje, vector)
            )

        def versor_pol(p, pol="V"):
            """
            Dado un vector de momento, y un índice de polarización,
            construye el correspondiente versor.
            """
            # Primero construyo la polarización vertical:
            """
            ptrans = np.sqrt(p[0] ** 2 + p[1] ** 2)
            versor = (
                np.array([0, 1, 0])
                if ptrans == 0.0
                else np.array([-p[1], p[0], 0]) / ptrans
            )
            """
            versor = np.array([0, 1, 0]) - p[1] * p / (
                p[0] ** 2 + p[1] ** 2 + p[2] ** 2
            )
            norm_versor = norm(versor)
            versor = versor / norm_versor if norm_versor else np.array([0.0, 0.0, 1.0])

            # Si la polarización es horizontal, lo roto 90º alrededor de p
            if pol == "H":
                versor = np.cross(versor, p)
            return versor

        en = norm(p)
        p = p / en
        # Construyo el vector de polarizacion.
        e_pol_in = versor_pol(p, pol)
        # Generar los ángulos
        theta_c, phi_c = sortear(en)
        # theta_c = PI / 2.0 + 0.0 * (theta_c - PI / 2.0)
        # Construyo el p de salida
        p_out = rotar(p, e_pol_in, theta_c)
        p_out = rotar(p_out, p, phi_c)
        # recalcular energia
        energia = en / (1.0 + en / 511 * (1 - np.cos(theta_c)))
        # Construyo el vector de polarización vertical para el p de salida
        e_pol_out = versor_pol(p_out)

        # Calculo la probabilidad de que el fotón saliente tenga
        # polarización vertical, como el cuadrado de la proyección entre
        # el versor de polarización entrante y el vector de polarización
        # saliente vertical.
        prob_V = (np.dot(e_pol_out, e_pol_in)) ** 2
        self.momento = p_out * energia
        self.energia = energia
        self.polarizacion = "V" if rng.uniform() < prob_V else "H"
        # print_debug("     listo", [self.momento, energia, self.polarizacion])

    def evol(self, t):
        """
        Evoluciona la posición del fotón (en cm) un tiempo t (en ns)
        """
        self.position = evol_photons(np.array([self.position]),
                               np.array([self.momento]),
                               np.array([self.energia]), t)[0]
        # print_debug(self)


class Evento:
    def __repr__(self):
        return (
            "..."
            + "\n".join(["\t" + foton.__repr__() for foton in self.fotones])
            + "\n"
            + "...\n"
        )

    def __init__(self, singlete=True, polarizado=None):
        enpair = 511

        # Foton de 1200
        # Una distribución uniforme sobre una esfera se
        # corresponde con la medida sin(theta)dtheta dphi
        # Para lograr la medida sin(theta)dtheta, transformamos
        # la distribución uniforme con un arccos:
        costheta_interval = CONO_FUENTE
        if EMITIR_EN_PLANO:
            arco_azimutal = tuple((-0.00001, 0.00001))
        else:
            arco_azimutal = tuple((-PI, PI))

        # Fotones de 511
        theta1 = np.arccos(rng.uniform(*costheta_interval))
        phi1 = rng.uniform(*arco_azimutal)
        p1 = enpair * np.array(
            [
                np.cos(phi1) * np.sin(theta1),
                np.sin(phi1) * np.sin(theta1),
                np.cos(theta1),
            ]
        )
        if singlete:
            self.fotones = [
                Foton(np.array([0, 0, 0]), p1, singlete=True),
                Foton(np.array([0, 0, 0]), -p1, singlete=True),
            ]
        elif polarizado:
            self.fotones = [
                Foton(np.array([0, 0, 0]), p1, polarizado),
                Foton(np.array([0, 0, 0]), -p1, polarizado),
            ]
        else:
            # Produce fotones con polarización predefinida
            self.fotones = [
                Foton(np.array([0, 0, 0]), p1, "V"),
                Foton(np.array([0, 0, 0]), -p1, "H"),
            ]
        if HERALDO:
            en0 = 1200
            theta0 = np.arccos(rng.uniform(*costheta_interval))
            phi0 = rng.uniform(*arco_azimutal)
            p0 = en0 * np.array(
                [
                    np.cos(phi0) * np.sin(theta0),
                    np.sin(phi0) * np.sin(theta0),
                    np.cos(theta0),
                ]
            )
            self.fotones.append(Foton(np.array([0, 0, 0]), p0, singlete=False))

    def draw_setup_2d(self, ax, color="yellow"):
        for foton in self.fotones:
            foton.draw_setup_2d(ax, color)

    def draw_setup_top(self, ax, color="yellow"):
        for foton in self.fotones:
            foton.draw_setup_top(ax, color)

    def evol(self, t):
        fotones_vivos = [foton for foton in self.fotones if (foton.energia and norm(foton.posicion) < 30.)]
        posiciones = np.array([foton.posicion for foton in fotones_vivos])
        momenta = np.array([foton.momento for foton in fotones_vivos])
        energias = np.array([foton.energia for foton in fotones_vivos])
        
        posiciones = evol_photons(posiciones, momenta, energias, t)                
        for foton, new_pos in zip(fotones_vivos, posiciones):
            foton.positicion = new_pos

        # print_debug("---")
        # Me quedo sólo con los fotones que están
        # a menos de 30 centimetros de la fuente.
        self.fotones = fotones_vivos


class Detector:
    """
    Representa las propiedades y el estado de un detector.
    """

    def __init__(
        self,
        eficiencia: float,
        posicion: tuple,
        radio: float = 5,
        upper: float = 511,
        retardo: float = 1.0,
    ):
        """
        eficiencia: la eficiencia del detector
        tipo: "start"/"stop"
        ret: retardo.
        posicion: coordenadas (en cm) relativas a la fuente.
        retardo: tiempo (en ns) de retardo entre la detección y
                 su registro.
        """
        self.eficiencia = eficiencia
        self.posicion = np.array(posicion)
        self.radio = radio
        self.retardo = retardo
        # Nivel representa la energía absorbida.
        self.nivel = 0.0
        self.upper = upper
        self.num_arribos = 0
        self.num_detecciones = 0
        self.ultima_deteccion = None
        self.estadistica_deteccion = []
        self._dt = -1
        self._p_deteccion = -1

    def __repr__(self):
        return f"""
        Detector:
            eficiencia={self.eficiencia}
            posicion={self.posicion}
            radio= {self.radio}
        """

    def draw_setup_2d(self, ax, color="green"):
        radio = self.radio
        x, y, z = self.posicion
        ax.add_patch(plt.Circle((x, z), radio, color=color))

    def draw_setup_top(self, ax, color="green"):
        radio = self.radio
        x, y, z = self.posicion
        ax.add_patch(plt.Circle((x, y), radio, color=color))

    def adentro(self, posicion):
        """
        Determina si posicion está en el interior del detector.
        """
        pos_rel = norm(self.posicion - posicion)
        return pos_rel < self.radio

    def check_eventos(self, dt: float, eventos: list):
        """
        Si uno de los fotones generado en alguno de los
        eventos pasa por el detector, se incrementa el
        nivel del detector.
        Finalmente, se devuelve el nivel del detector, y
        se resetea el nivel a 0.
        """
        # Si cambiamos dt desde la última vez,
        # recalculamos la probabilidad de
        # detección en función de la eficiencia.
        if self._dt != dt:
            p_deteccion = prob_detection(self.eficiencia, 2 * self.radio / (29.9 * dt))
            print("p_deteccion:", p_deteccion)
            self._p_deteccion = p_deteccion
            self._dt = dt
        else:
            p_deteccion = self._p_deteccion

        # Cada evento tiene tres fotones: primero el de 1200
        # y luego el par de 511.    
        fotones_activos = [foton  for evento in eventos for foton in evento.fotones if foton.energia]
        posiciones = np.array([foton.posicion for foton in fotones_activos])
        indices = dentro_detector(posiciones, self.posicion, self.radio)
        uniform = rng.uniform
        for indice in indices:
            foton = fotones_activos[indice]
            self.num_arribos += 1
            if uniform() < p_deteccion:
                self.nivel += foton.energia
                # El fotón se "absorbe" sacándolo
                # del sistema.
                foton.aniquilar()

        if self.nivel > 0:
            # print_debug("nivel >0")
            nivel = self.nivel
            self.nivel = 0.0
            # Esto simula el discriminador: si la energia es >511,
            # sólo acepta la señal con probabilidad (511/nivel)**2
            result = (self.upper / nivel) ** 2 > rng.uniform()
            # print_debug("Nivel del detector: ", nivel, "->", result)
            self.num_detecciones += 1
            return result


class Dispersor:
    """
    Representa las propiedades y el estado de un detector.
    """
    
    def __init__(
        self, lambda_dispersion: float = 1.0, lambda_absorcion: float = 1000000.0
    ):
        """
        lambda_dispersion: longitud de penetración asociada
                            a la absorcion (1/ (rho * sigma_dis), en [cm])
        lambda_absorcion: longitud de penetración asociada
                          a la absorcion (1/ (rho * sigma_abs),  en [cm])
        posicion: coordenadas del centro del cilindro
        alto: alto del cilindro
        radio: radio del cilindro
        """
        self.lambda_dispersion = lambda_dispersion
        self.lambda_absorcion = lambda_absorcion
        self.cuenta_compton = 0
        self.cuenta_absorcion = 0

    def adentro(self, posicion):
        """
        Determina si posicion está en el interior del dispersor(cilindrico).
        """
        raise NotImplementedError

    def check_scattering(self, dt: float, eventos: list):
        """
        Simula la interacción de los fotones en el sistema
        con el blanco, en el intervalo de tiempo dt.
        La interacción sólo es posible si el fotón se
        encuentra dentro del blanco.

        """
        probabilidad_dispersion = (
            1.0 - np.exp(-29.9 * dt / self.lambda_dispersion)
            if self.lambda_dispersion
            else 1.0
        )
        probabilidad_absorcion = (
            1.0 - np.exp(-29.9 * dt / self.lambda_absorcion)
            if self.lambda_absorcion
            else 1.0
        )

        fotones_in = [(evento, foton,) for evento in eventos for foton in evento.fotones if foton.energia]

        if isinstance(self, BlancoCilindrico):
            posiciones = np.array([foton.posicion for evento, foton in fotones_in])
            fotones_in = [fotones_in[i] for i in
                          dentro_cilindro(posiciones, self.posicion, self.radio, self.alto)]
        elif isinstance(self, ShieldCilindrico):
            posiciones = np.array([foton.posicion for evento, foton in fotones_in])
            fotones_in = [fotones_in[i] for i in
                          dentro_anillo(posiciones, self.posicion, self.radio_interior, self.radio_exterior, self.alto, {"x":0,"y":1,"z":2}[self.eje])]
        else:
            fotones_in = [(evento, foton,) for evento, foton in fotones_in if self.adentro(foton)]
        
        for evento, foton in fotones_in:
            # Simula la absorción
            if rng.uniform() < probabilidad_absorcion:
                # # print_debug("                 absorcion!")
                foton.aniquilar()
                self.cuenta_absorcion += 1

            elif rng.uniform() < probabilidad_dispersion:
                # print_debug("                 dispersion!")
                
                if foton.singlete:
                    # print_debug("Estoy en un singlete!")
                    # Como lo detectamos, definimos su
                    # polarizacion, y la de su gemelo.
                    foton.polarizacion = "V" if rng.choice(2, 1)[0] else "H"
                    foton.singlete = False
                    for candidato_gemelo in evento.fotones:
                        if candidato_gemelo.singlete:
                            candidato_gemelo.singlete = False
                            candidato_gemelo.polarizacion = (
                                "H" if foton.polarizacion == "V" else "H"
                            )

                # Cambiamos la dirección con una distribución
                # de acuerdo a Klein-Nishina
                foton.compton()
                self.cuenta_compton += 1


class BlancoCilindrico(Dispersor):

    lista_adentro = dentro_cilindro
    def __init__(
        self,
        lambda_dispersion: float = 1.0,
        lambda_absorcion: float = 1000000.0,
        posicion: tuple = np.array([0, 0, 10]),
        alto: float = 1,
        radio: float = 0.5,
        color="cyan",
    ):
        """
        lambda_dispersion: longitud de penetracion asociada
                            a la dispersion (cm)
        lambda_absorcion: longitud de penetracion asociada
                          a la absorcion (cm)
        posicion: coordenadas del centro del cilindro
        alto: alto del cilindro
        radio: radio del cilindro
        """
        super().__init__(lambda_dispersion, lambda_absorcion)
        self.posicion = np.array(posicion)
        self.radio = radio
        self.alto = alto
        self.color = color

    def __repr__(self):
        return "* Blanco Cilíndrico:\n" + "\n\t".join(
            [
                f"lambda_dispersion={self.lambda_dispersion}",
                f"lambda_absorcion={self.lambda_absorcion}",
                f"posicion={self.posicion}",
                f"dimensiones= {self.radio}x{self.alto}",
            ]
        )

    def draw_setup_2d(self, ax, color=None):
        if color is None:
            color = self.color

        x, y, z = self.posicion
        r = self.radio
        alto = self.alto
        ax.add_patch(plt.Rectangle((x - r, z - 0.5 * alto), 2 * r, alto, color=color))

    def draw_setup_top(self, ax, color=None):
        if color is None:
            color = self.color

        x, y, z = self.posicion
        r = self.radio
        ax.add_patch(plt.Circle((x, y), r, color=color))

    def adentro(self, posicion):
        """
        Determina si posicion está en el interior del dispersor(cilindrico).
        """
        pos_rel = self.posicion - posicion
        if abs(pos_rel[2]) > 0.5 * self.alto:
            return False
        if pos_rel[0] ** 2 + pos_rel[1] ** 2 > self.radio**2:
            return False
        return True


class ShieldCilindrico(Dispersor):
    def __init__(
        self,
        lambda_dispersion: float = 1000000.0,
        lambda_absorcion: float = 1.0,
        posicion: tuple = np.array([0.0, 0.0, 0.0]),
        alto: float = 2.0,
        radio_interior: float = 1.5,
        radio_exterior: float = 0.5,
        eje: str = "z",
    ):
        """
        Representa un cilindro hueco, de radio interior
        ``radio_interior``, radio exterior ``radio_exterior``,
        y altura ``alto``, con su centro localizado en
        ``posicion`` y con su eje de simetría alineado con el
        eje `eje`.

        cross_section: probabilidad de dispersion
        absorcion: probabilidad de absorcion
        posicion: coordenadas del centro del cilindro
        alto: alto del cilindro
        radio: radio del cilindro
        """
        super().__init__(lambda_dispersion, lambda_absorcion)
        self.posicion = np.array(posicion)
        self.radio_interior = radio_interior
        self.radio_exterior = radio_exterior
        self.alto = alto
        self.eje = eje

    def __repr__(self):
        return "* Shielding:\n" + "\n\t".join(
            [
                f"lambda_dispersion={self.lambda_dispersion}",
                f"lambda_absorcion={self.lambda_absorcion}",
                f"posicion={self.posicion}",
                (
                    "dimensiones="
                    + f"({self.radio_exterior}-{self.radio_interior})"
                    + f"x{self.alto}"
                ),
            ]
        )

    def draw_setup_2d(self, ax, color="lightgray"):
        r_e, r_i = self.radio_exterior, self.radio_interior
        x, y, z = self.posicion
        alto = self.alto
        ancho = r_e - r_i
        if self.eje == "z":
            ax.add_patch(
                plt.Rectangle((x - r_e, z - 0.5 * alto), ancho, alto, color=color)
            )
            ax.add_patch(
                plt.Rectangle((x + r_i, z - 0.5 * alto), ancho, alto, color=color)
            )
        elif self.eje == "x":
            ax.add_patch(
                plt.Rectangle((z - 0.5 * alto, x - r_e), alto, ancho, color=color)
            )
            ax.add_patch(
                plt.Rectangle((z - 0.5 * alto, x + r_i), alto, ancho, color=color)
            )
        elif self.eje == "y":
            ax.add_patch(plt.Circle((x, y), r_e, color=color))
            ax.add_patch(plt.Circle((x, y), r_i, color="white"))

    def draw_setup_top(self, ax, color="lightgray"):
        r_e, r_i = self.radio_exterior, self.radio_interior
        x, y, z = self.posicion
        alto = self.alto
        ancho = r_e - r_i
        if self.eje == "y":
            ax.add_patch(
                plt.Rectangle((x - r_e, z - 0.5 * alto), ancho, alto, color=color)
            )
            ax.add_patch(
                plt.Rectangle((x + r_i, z - 0.5 * alto), ancho, alto, color=color)
            )
        elif self.eje == "x":
            ax.add_patch(
                plt.Rectangle((z - 0.5 * alto, x - r_e), alto, ancho, color=color)
            )
            ax.add_patch(
                plt.Rectangle((z - 0.5 * alto, x + r_i), alto, ancho, color=color)
            )
        elif self.eje == "z":
            ax.add_patch(plt.Circle((x, y), r_e, color=color))
            ax.add_patch(plt.Circle((x, y), r_i, color="white"))

    def adentro(self, posicion):
        """
        Determina si posicion está en el interior
        del dispersor(cilindrico).
        """
        pos_rel = self.posicion - posicion
        if self.eje == "x":
            pos_rel[0], pos_rel[2] = pos_rel[2], pos_rel[0]
        elif self.eje == "y":
            pos_rel[1], pos_rel[2] = pos_rel[2], pos_rel[1]

        if abs(pos_rel[2]) > 0.5 * self.alto:
            return False

        r1sq, r2sq = self.radio_interior**2, self.radio_exterior**2
        return r2sq > pos_rel[0] ** 2 + pos_rel[1] ** 2 > r1sq


class Experimento:
    def __init__(
        self,
        duracion: float = 80,  # ns
        step: float = 0.001,  # ns
        ventana: float = 50,  # ns
        canales: int = 2048,  # cantidad de canales
        flujo_eventos: float = 0.1,  # eventos / ns
        polarizacion_fuente=None,  # Asumir que los fotones salen polarizados
        # Los detectores
        start: Detector = Detector(
            eficiencia=0.9,
            posicion=np.array([10, 0, 10]),
            radio=2,
            retardo=0,
        ),
        stop: Detector = Detector(
            eficiencia=0.6,
            posicion=np.array([-10, 0, -10]),
            radio=2,
            retardo=1,
        ),
        # Los Blancos
        blancos: list = [],
        folder=None,  # Generar un nombre si no se pasa como argumento.
        checkpoint_interval=60,  # seg
    ):

        self.duracion = duracion
        self.step = step
        self.ventana = ventana
        self.canales = canales
        self.eventos = []
        self.flujo_eventos = flujo_eventos
        self.blancos = blancos
        self.detector_start = start
        self.detector_stop = stop
        self.coincidencias = np.array([0.0 for k in range(canales)])
        self.clicks_start = []
        self.clicks_stop = []
        self.time = 0
        self.folder = folder
        self.checkpoint_interval = checkpoint_interval
        self.polarizacion_fuente = polarizacion_fuente
        print("Checkpoint interval: ", self.checkpoint_interval)

    def __repr__(self):
        return (
            f"""
        Start: {self.detector_start}
        Stop: {self.detector_stop}
        Blancos:
        """
            + "\n".join(f"\n{b}" for b in self.blancos)
            + f"\nCoincidencias: {self.coincidencias}"
        )

    def draw_setup_2d(self, ax):
        # Dibuja los detectores
        legend = (
            f"t={time_with_units(self.time)}\n"
            f"flujo={self.flujo_eventos}\n"
            f"cuentas={sum(self.coincidencias)}\n"
        )
        if self.clicks_start:
            legend += (
                f"start: {len(self.clicks_start)} cuentas"
                f" entre  {self.clicks_start[-1]} y"
                f" {self.clicks_start[0]}\n"
            )
        else:
            legend += "nada en starts\n"
        if self.clicks_stop:
            legend += (
                f"stop: {len(self.clicks_stop)} cuentas"
                f" entre  {self.clicks_stop[-1]} y"
                f" {self.clicks_stop[0]}\n"
            )
        else:
            legend += "nada en stop"

        ax.text(-15, 10, legend)
        self.detector_start.draw_setup_2d(ax)
        self.detector_stop.draw_setup_2d(ax)
        # Dibuja los dispersores y shieldings
        for blanco in self.blancos:
            blanco.draw_setup_2d(ax)
        # Dibuja los fotones
        for evento in self.eventos:
            evento.draw_setup_2d(ax)

    def draw_setup_top(self, ax):
        # Dibuja los detectores
        legend = (
            f"t={time_with_units(self.time)}\n"
            f"flujo={self.flujo_eventos}\n"
            f"cuentas={sum(self.coincidencias)}\n"
        )
        if self.clicks_start:
            legend += (
                f"start: {len(self.clicks_start)} cuentas"
                f" entre  {self.clicks_start[-1]} y {self.clicks_start[0]}\n"
            )
        else:
            legend += "nada en starts\n"
        if self.clicks_stop:
            legend += (
                f"stop: {len(self.clicks_stop)} cuentas entre"
                f"  {self.clicks_stop[-1]} y {self.clicks_stop[0]}\n"
            )
        else:
            legend += "nada en stop"

        ax.text(-15, 10, legend)
        self.detector_start.draw_setup_top(ax)
        self.detector_stop.draw_setup_top(ax)
        # Dibuja los dispersores y shieldings
        for blanco in self.blancos:
            blanco.draw_setup_top(ax)
        # Dibuja los fotones
        for evento in self.eventos:
            evento.draw_setup_top(ax)

    def check_coincidencias(self):
        """
        Avanza el estado de las señales, acumula coincidencias,
        y limpia los clicks perdidos.

        Cuando uno de los detectores registra un evento, (en self.evol)
        se genera una nueva entrada en la cola correspondiente (start o stop)
        y se inicializa con (menos) el retardo correspondiente.
        En cada paso, cada elemento de la cola se actualiza en un paso de valor
        `self.step`. Eventualmente, estos valores se volverán
        positivos, lo que representa una señal que efectivamente llegó a
        la placa de coincidencias.

        Si una señal de stop está "activa", y no hay una señal de "start"
        previa, la señal se descarta.
        Si la señal de "start" estuvo viva por un tiempo mayor a la ventana,
        entonces se descarta.
        Si a la vez están activas una señal de start y otra de stop, y la de
        stop es posterior al start, se calcula la diferencia de tiempo.
        Si es menor al tamaño de la ventana, se guarda, si no, se descarta.
        """
        # Avanza el estado de la señal y acumula coincidencias.
        step = self.step
        start_q = self.clicks_start
        stop_q = self.clicks_stop
        # if start_q:
        #    # print_debug("    cola start", (start_q[0], start_q[-1]))
        # if stop_q:
        #    # print_debug("    cola stop", (stop_q[0], stop_q[-1]))

        # La cola está vacía. Nada para hacer.
        if not start_q and not stop_q:
            return

        for i, val in enumerate(start_q):
            start_q[i] += step
        for i, val in enumerate(stop_q):
            stop_q[i] += step

        # Remuevo los starts que se salieron de la ventana.
        while start_q and start_q[0] > self.ventana:
            # old =
            start_q.pop(0)
            # print_debug(old, "ya no es un start valido")

        # Remuevo los stops con tiempo positivo, que son más
        # viejos que el start vigente.
        if start_q:
            while stop_q and stop_q[0] > 0 and start_q[0] < stop_q[0]:
                # old =
                stop_q.pop(0)
                # print_debug(
                #    old, "ya no es un stop valido, ya que ", start_q[0], "es más nuevo"
                #)
        else:
            # Si no hay nada en la cola de start,
            # remover los stops con tiempos positivos
            while stop_q and stop_q[0] > 0:
                stop_q.pop(0)

        if stop_q and start_q and stop_q[0] > 0:
            # registramos un dt igual a la diferencia entre el stop y el start,
            # mas un ruido gaussiano con un ancho de 0.1ns (de acuerdo con
            # la incerteza estimada para la electrónica de coincidencias.
            older_stop = stop_q.pop(0)
            older_start = start_q.pop(0)
            dt = older_start - older_stop  # + 0.1 * rng.normal()
            canal = int(dt / self.ventana * self.canales)
            # Registro la coincidencia.
            try:
                # print_debug("Se ha formado una pareja!", [dt, canal])
                self.coincidencias[canal] += 1.0
            except IndexError:
                pass
                # print_debug("  demasiado tarde...")

            # Remuevo los starts que son más viejos que el actual
            # stop, ya que no podrían haberse detectado.
            while start_q and start_q[0] > older_stop:
                start_q.pop(0)

        assert self.clicks_start is start_q
        assert self.clicks_stop is stop_q

    def check_point(self, folder=None, show_plots=False):
        print(datetime.now().strftime("%H:%M:%S"))
        if folder is None:
            folder = self.folder

        with open(f"{folder}/checkpoint.pkl", "wb") as f_out:
            pickle.dump(self, f_out)

        with open(f"{folder}/cuentas.txt", "w") as f_out:
            f_out.write(str(self.time))
            for c in self.coincidencias:
                f_out.write("\n" + str(c))

        print("cuentas:", sum(self.coincidencias))
        for b in self.blancos:
            print(b, " tuvo ", b.cuenta_compton, " eventos Compton y")
            print("         ", b.cuenta_absorcion, " absorciones")

        for name, d in [("start", self.detector_start), ("stop", self.detector_stop)]:
            print(" El detector ", name, " tuvo ", d.num_arribos, " arribos y ")
            print("         ", d.num_detecciones, " detecciones")
            n_det = len(d.estadistica_deteccion)
            if n_det:
                tdet = sum(d.estadistica_deteccion) / n_det
                disp_tdet = (
                    sum([t**2 for t in d.estadistica_deteccion]) / n_det - tdet**2
                ) ** 0.5
                print("   tiempo medio entre detecciones:", tdet, "+/-", disp_tdet)

        fig, ax = plt.subplots()
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        self.draw_setup_2d(ax)
        plt.savefig(f"{folder}/setup.png")
        if show_plots:
            print("show!")
            plt.show()
        else:
            plt.close()

        print("draw top")
        fig, ax = plt.subplots()
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        self.draw_setup_top(ax)
        plt.savefig(f"{folder}/setuptop.png")
        if show_plots:
            print("show!")
            plt.show()
        else:
            plt.close()

        fig, ax = plt.subplots()
        plt.plot(self.coincidencias)
        plt.savefig(f"{folder}/coincidencias.png")
        if show_plots:
            print("show!")
            plt.show()
        else:
            plt.close()

    def evol(self):
        """
        realiza un paso de evolución
        """
        # Mueve todos los fotones
        step = self.step

        for evento in self.eventos:
            evento.evol(step)
        # Elimina todos los fotones que se escaparon
        self.eventos = [ev for ev in self.eventos if ev.fotones]
        # Actualiza la cola de coincidencias.
        self.check_coincidencias()

        # Simula la dispersión con los blancos
        # Si se produce una dispersión, cambia el momento
        # Si se produce absorción, aniquila al fotón.
        for blanco in self.blancos:
            blanco.check_scattering(step, self.eventos)

        # Revisa el estado de los detectores, y registra los clicks.
        # Detector.check_eventos devuelve un número en función de
        # la energía absorbida. Podemos usar eso como criterio para
        # filtrar (esto es, simular los discriminadores).
        for detector_name, detector, clicks in [
            ("start", self.detector_start, self.clicks_start),
            ("stop", self.detector_stop, self.clicks_stop),
        ]:

            if detector.check_eventos(step, self.eventos):
                clicks.append(-detector.retardo)
                # print_debug(detector_name, " hizo click!", self.time)
                last_detec = detector.ultima_deteccion
                if last_detec is not None:
                    tiempo_entre_clicks = self.time - last_detec
                    detector.estadistica_deteccion.append(tiempo_entre_clicks)
                detector.ultima_deteccion = self.time

    def simular(self, duracion=None):
        if duracion is None:
            duracion = self.duracion
        step = self.step
        print("simular  ", duracion, "ns")
        # print_debug("duracion:", duracion)
        prob_emision = self.flujo_eventos * step
        # print_debug("probabilidad emision:", prob_emision)

        time_print = time()
        time_checkpoint = time()
        while duracion > 0:
            real_time = time()
            if (real_time - time_print) > 5:
                print(datetime.now().strftime("%H:%M:%S"))
                print(
                    self.time,
                    "eventos:",
                    len(self.eventos),
                    "clicks start:",
                    len(self.clicks_start),
                    "clicks stop:",
                    len(self.clicks_stop),
                )
                time_print = real_time

            if (real_time - time_checkpoint) > self.checkpoint_interval:
                print("Check point ")
                self.check_point()
                time_checkpoint = real_time

            duracion -= step
            self.time += step

            if self.polarizacion_fuente:
                parms_evento = {
                    "singlete": False,
                    "polarizado": self.polarizacion_fuente,
                }
            else:
                parms_evento = {"singlete": True, "polarizado": None}

            if prob_emision < 1:
                if rng.uniform() < prob_emision:
                    self.eventos.append(Evento(**parms_evento))
            else:
                for k in range(int(prob_emision)):
                    self.eventos.append(Evento(**parms_evento))
            self.evol()
