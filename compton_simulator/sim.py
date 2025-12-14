from datetime import datetime
from time import time
import numpy as np
from numpy.random import default_rng
from matplotlib import pyplot as plt
import pickle

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
    print_debug("Calculando la probabilidad de deteccion")

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
    cos_theta = KN_CosTheta(en)
    theta = np.arccos(cos_theta)
    PI = np.pi

    def dof_phi(phi, en):
        la = 1 / (1 + en * (1 - cos_theta))
        a = 1 - cos_theta**2
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
            print_debug(f"  convergence achieved in {20-rem_steps}")
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
            print_debug(f"  convergence achieved in {20-rem_steps}")
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
        print_debug("   Dispersion compton")
        p, pol = self.momento, self.polarizacion
        if pol is None:
            pol = "V" if rng.uniform() < 0.5 else "H"

        print_debug("Polarizacion: ", pol)

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
        print_debug("     listo", [self.momento, energia, self.polarizacion])

    def evol(self, t):
        """
        Evoluciona la posición del fotón (en cm) un tiempo t (en ns)
        """
        if self.energia:
            self.posicion = self.posicion + t * 29.9 * self.momento / self.energia
        print_debug(self)


class Evento:
    def __repr__(self):
        1 / 0
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

    def draw_setup_2d(self, ax):
        for foton in self.fotones:
            foton.draw_setup_2d(ax)

    def draw_setup_top(self, ax):
        for foton in self.fotones:
            foton.draw_setup_top(ax)

    def evol(self, t):
        for foton in self.fotones:
            foton.evol(t)
        print_debug("---")
        # Me quedo sólo con los fotones que están
        # a menos de 30 centimetros de la fuente.
        self.fotones = [foton for foton in self.fotones if norm(foton.posicion) < 30]


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

        for evento in eventos:
            # Cada evento tiene tres fotones: primero el de 1200
            # y luego el par de 511.
            for foton in evento.fotones:
                # Podríamos agregar también eficiencias
                # en función de la energia
                if self.adentro(foton.posicion):
                    self.num_arribos += 1
                    if rng.uniform() < p_deteccion:
                        self.nivel += foton.energia
                        # El fotón se "absorbe" sacándolo
                        # del sistema.
                        foton.aniquilar()

        if self.nivel > 0:
            print_debug("nivel >0")
            nivel = self.nivel
            self.nivel = 0.0
            # Esto simula el discriminador: si la energia es >511,
            # sólo acepta la señal con probabilidad (511/nivel)**2
            result = (self.upper / nivel) ** 2 > rng.uniform()
            print_debug("Nivel del detector: ", nivel, "->", result)
            self.num_detecciones += 1
            return result


class Dispersor:
    """
    Representa las propiedades y el estado de un detector.
    """

    def __init__(self,
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
        for evento in eventos:
            # Cada evento tiene tres fotones: primero el de 1200
            # y luego el par de 511.
            for foton in evento.fotones:
                # Podríamos agregar también eficiencias
                # en función de la energia
                if not self.adentro(foton.posicion):
                    continue

                # Simula la absorción
                if rng.uniform() < probabilidad_absorcion:
                    # print_debug("                 absorcion!")
                    foton.aniquilar()
                    self.cuenta_absorcion += 1

                elif rng.uniform() < probabilidad_dispersion:
                    print_debug("                 dispersion!")

                    if foton.singlete:
                        print_debug("Estoy en un singlete!")
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


# Numba-friendly helper: check if a point is inside an annulus (ring)
# oriented along a numeric axis. This avoids dynamic numpy indexing
# inside Numba-compiled code by using explicit component access.
@numba.njit
def dentro_anillo(pos, centro, r_in, r_out, axis):
    # pos and centro expected to be 1D arrays of length 3
    # axis is an integer: 0 -> x, 1 -> y, 2 -> z
    if axis == 0:
        dx = pos[1] - centro[1]
        dy = pos[2] - centro[2]
    elif axis == 1:
        dx = pos[0] - centro[0]
        dy = pos[2] - centro[2]
    else:
        dx = pos[0] - centro[0]
        dy = pos[1] - centro[1]

    dist2 = dx * dx + dy * dy
    return (dist2 >= r_in * r_in) and (dist2 <= r_out * r_out)


class ShieldCilindrico(Dispersor):
    """
    Cilindro anular (caja cilíndrica) que puede orientarse en el eje x/y/z.
    Accepts eje as an int (0/1/2) or a string ('x','y','z').
    """

    def __init__(
        self,
        centro=(0.0, 0.0, 0.0),
        radio_interno=0.0,
        radio_externo=5.0,
        alto=10.0,
        eje='z',
        lambda_dispersion: float = 1.0,
        lambda_absorcion: float = 1000000.0,
    ):
        super().__init__(lambda_dispersion, lambda_absorcion)
        self.centro = np.array(centro, dtype=np.float64)
        self.radio_interno = float(radio_interno)
        self.radio_externo = float(radio_externo)
        self.alto = float(alto)

        # Accept numeric or string eje; store numeric axis (0,1,2)
        if isinstance(eje, str):
            eje_l = eje.lower()
            if eje_l == 'x':
                axis = 0
            elif eje_l == 'y':
                axis = 1
            elif eje_l == 'z':
                axis = 2
            else:
                raise ValueError(f"eje string must be 'x','y' or 'z', got {eje}")
        else:
            axis = int(eje)
            if axis not in (0, 1, 2):
                raise ValueError(f"eje int must be 0,1 or 2, got {eje}")

        self.eje = axis

    def adentro(self, posicion):
        """
        Determina si posicion está en el interior del cilindro anular.
        Uses the numba-compiled dentro_anillo for the transverse check and a
        simple axial height check for the longitudinal coordinate.
        """
        pos_arr = np.asarray(posicion, dtype=np.float64)
        # axial coordinate
        coord = pos_arr[self.eje]
        centro_coord = self.centro[self.eje]
        if abs(coord - centro_coord) > (self.alto / 2.0):
            return False
        # transverse annulus check using numeric axis
        return bool(dentro_anillo(pos_arr, self.centro, self.radio_interno, self.radio_externo, self.eje))
