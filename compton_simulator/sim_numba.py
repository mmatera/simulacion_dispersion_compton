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

@numba.njit
def eficiencia_media_numba(p, n):
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

@numba.njit
def prob_detection_numba(eficiencia, n):
    if eficiencia > 10.0:
        return 1.0
    steps = 20
    y0 = eficiencia
    p = eficiencia
    while steps:
        y, dy = eficiencia_media_numba(p, n)
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
        steps -= 1
    return p

@numba.njit
def dof_costheta_numba(costheta, en):
    if en == 0:
        return 3 / 8 * (1 + costheta**2)
    else:
        la = 1 / (1 + en * (1 - costheta))
        val = en**3 * la**2 * (la + 1 / la - (1 - costheta**2))
        w = 1 + 2 * en
        norm = 2 * en * (2 + en * (1 + en) * (8 + en)) / (w**2)
        norm += np.log(w) * (en * (en - 2) - 2)
        return val / norm

@numba.njit
def cof_costheta_numba(costheta, en):
    if en == 0:
        return (4 + 3 * costheta + costheta**3) / 8.0
    else:
        total = 0.0
        for i in range(100):
            u = -1.0 + i * (costheta+1.0)/99.0
            total += dof_costheta_numba(u, en)
        return 0.01 * total * (1.0 + costheta)

def KN_CosTheta(en=1):
    y0 = rng.uniform()
    x0 = 2.0 * y0 - 1.0
    rem_steps = 20
    while rem_steps:
        y1 = cof_costheta_numba(x0, en)
        dy = y0 - y1
        dx = dy / dof_costheta_numba(x0, en)
        if abs(dx) < 0.01:
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

@numba.njit
def dof_phi_numba(phi, cos_theta, en):
    la = 1 / (1 + en * (1 - cos_theta))
    a = 1 - cos_theta**2
    w = a / (la + 1.0 / la - a)
    return (1 + w * np.cos(2 * phi)) / (2.0 * PI)

@numba.njit
def cof_phi_numba(phi, cos_theta, en):
    la = 1 / (1 + en * (1 - cos_theta))
    a = 1 - cos_theta**2
    w = a / (la + 1.0 / la - a)
    u = 2 * phi
    return 0.5 + 0.25 * (u + w * np.sin(u)) / PI

def KN_ThetaPhi(en=1):
    cos_theta = KN_CosTheta(en)
    theta = np.arccos(cos_theta)
    y0 = rng.uniform()
    x0 = (2.0 * y0 - 1.0) * PI
    rem_steps = 20
    while rem_steps:
        y1 = cof_phi_numba(x0, cos_theta, en)
        dy = y0 - y1
        dx = dy / dof_phi_numba(x0, cos_theta, en)
        if abs(dx) < 0.01:
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

@numba.njit
def rotar_numba(vector, eje, angulo):
    doteje = np.dot(vector, eje)
    cross1 = np.cross(eje, np.cross(eje, vector))
    cross2 = np.cross(eje, vector)
    return doteje * eje - np.cos(angulo) * cross1 + np.sin(angulo) * cross2

@numba.njit
def versor_pol_numba(p, pol_v):
    versor = np.array([0.0, 1.0, 0.0]) - p[1] * p / (p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
    norm_versor = np.sqrt(np.sum(versor ** 2))
    if norm_versor:
        versor = versor / norm_versor
    else:
        versor = np.array([0.0, 0.0, 1.0])
    # pol_v: 0 for "V", 1 for "H"
    if pol_v == 1:
        versor = np.cross(versor, p)
    return versor

def pol_to_int(pol):
    return 0 if pol == "V" else 1

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
        color = (1.0, min(1.0, self.energia / 511), max(0, 1 - 511 / self.energia))
        historia_x = [p[0] for p in self.historia] + [x]
        historia_y = [p[1] for p in self.historia] + [y]

        ax.scatter([x], [y], color=color)
        ax.plot(historia_x, historia_y, color=color)

    def draw_setup_2d(self, ax, color="yellow"):
        if self.energia == 0:
            return

        x, y, z = self.posicion
        if abs(y) > 1:
            return
        px, py, pz = self.momento / self.energia
        assert abs(pz) < 2 and abs(px) < 2, [self.energia, self.momento]
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
        p, pol = self.momento, self.polarizacion
        if pol is None:
            pol = "V" if rng.uniform() < 0.5 else "H"
        en = norm(p)
        p_normed = p / en
        pol_v = pol_to_int(pol)
        e_pol_in = versor_pol_numba(p_normed, pol_v)
        theta_c, phi_c = KN_ThetaPhi(en)
        p_out = rotar_numba(p_normed, e_pol_in, theta_c)
        p_out = rotar_numba(p_out, p_normed, phi_c)
        energia = en / (1.0 + en / 511 * (1 - np.cos(theta_c)))
        e_pol_out = versor_pol_numba(p_out, 0)
        prob_V = (np.dot(e_pol_out, e_pol_in)) ** 2
        self.momento = p_out * energia
        self.energia = energia
        self.polarizacion = "V" if rng.uniform() < prob_V else "H"

    def evol(self, t):
        if self.energia:
            self.posicion = self.posicion + t * 29.9 * self.momento / self.energia

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
        costheta_interval = CONO_FUENTE
        if EMITIR_EN_PLANO:
            arco_azimutal = tuple((-0.00001, 0.00001))
        else:
            arco_azimutal = tuple((-PI, PI))

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
        for foton in self.fotones:
            foton.evol(t)
        self.fotones = [foton for foton in self.fotones if norm(foton.posicion) < 30]

class Detector:
    def __init__(
        self,
        eficiencia: float,
        posicion: tuple,
        radio: float = 5,
        upper: float = 511,
        retardo: float = 1.0,
    ):
        self.eficiencia = eficiencia
        self.posicion = np.array(posicion)
        self.radio = radio
        self.retardo = retardo
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
        pos_rel = norm(self.posicion - posicion)
        return pos_rel < self.radio

    def check_eventos(self, dt: float, eventos: list):
        if self._dt != dt:
            self._p_deteccion = prob_detection_numba(self.eficiencia, 2 * self.radio / (29.9 * dt))
            self._dt = dt
        p_deteccion = self._p_deteccion

        for evento in eventos:
            for foton in evento.fotones:
                if self.adentro(foton.posicion):
                    self.num_arribos += 1
                    if rng.uniform() < p_deteccion:
                        self.nivel += foton.energia
                        foton.aniquilar()

        if self.nivel > 0:
            nivel = self.nivel
            self.nivel = 0.0
            result = (self.upper / nivel) ** 2 > rng.uniform()
            self.num_detecciones += 1
            return result

class Dispersor:
    def __init__(
        self, lambda_dispersion: float = 1.0, lambda_absorcion: float = 1000000.0
    ):
        self.lambda_dispersion = lambda_dispersion
        self.lambda_absorcion = lambda_absorcion
        self.cuenta_compton = 0
        self.cuenta_absorcion = 0

    def adentro(self, posicion):
        raise NotImplementedError

    def check_scattering(self, dt: float, eventos: list):
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
            for foton in evento.fotones:
                if not self.adentro(foton.posicion):
                    continue
                if rng.uniform() < probabilidad_absorcion:
                    foton.aniquilar()
                    self.cuenta_absorcion += 1
                elif rng.uniform() < probabilidad_dispersion:
                    if foton.singlete:
                        foton.polarizacion = "V" if rng.choice(2, 1)[0] else "H"
                        foton.singlete = False
                        for candidato_gemelo in evento.fotones:
                            if candidato_gemelo.singlete:
                                candidato_gemelo.singlete = False
                                candidato_gemelo.polarizacion = (
                                    "H" if foton.polarizacion == "V" else "H"
                                )
                    foton.compton()
                    self.cuenta_compton += 1

class BlancoCilindrico(Dispersor):
    def __init__(
        self,
        lambda_dispersion: float = 1.0,
        lambda_absorcion: float = 1000000.0,
        posicion: tuple = np.array([0, 0, 10]),
        alto: float = 1,
        radio: float = 0.5,
        color="cyan",
    ):
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
        duracion: float = 80,
        step: float = 0.001,
        ventana: float = 50,
        canales: int = 2048,
        flujo_eventos: float = 0.1,
        polarizacion_fuente=None,
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
        blancos: list = [],
        folder=None,
        checkpoint_interval=60,
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
        for blanco in self.blancos:
            blanco.draw_setup_2d(ax)
        for evento in self.eventos:
            evento.draw_setup_2d(ax)

    def draw_setup_top(self, ax):
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
        for blanco in self.blancos:
            blanco.draw_setup_top(ax)
        for evento in self.eventos:
            evento.draw_setup_top(ax)

    def check_coincidencias(self):
        step = self.step
        start_q = self.clicks_start
        stop_q = self.clicks_stop

        if not start_q and not stop_q:
            return

        for i in range(len(start_q)):
            start_q[i] += step
        for i in range(len(stop_q)):
            stop_q[i] += step

        while start_q and start_q[0] > self.ventana:
            start_q.pop(0)

        if start_q:
            while stop_q and stop_q[0] > 0 and start_q[0] < stop_q[0]:
                stop_q.pop(0)
        else:
            while stop_q and stop_q[0] > 0:
                stop_q.pop(0)

        if stop_q and start_q and stop_q[0] > 0:
            older_stop = stop_q.pop(0)
            older_start = start_q.pop(0)
            dt = older_start - older_stop
            canal = int(dt / self.ventana * self.canales)
            try:
                self.coincidencias[canal] += 1.0
            except IndexError:
                pass
            while start_q and start_q[0] > older_stop:
                start_q.pop(0)

    def check_point(self, folder=None, show_plots=False):
        if folder is None:
            folder = self.folder
        with open(f"{folder}/checkpoint.pkl", "wb") as f_out:
            pickle.dump(self, f_out)
        with open(f"{folder}/cuentas.txt", "w") as f_out:
            f_out.write(str(self.time))
            for c in self.coincidencias:
                f_out.write("\n" + str(c))
        fig, ax = plt.subplots()
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        self.draw_setup_2d(ax)
        plt.savefig(f"{folder}/setup.png")
        plt.close()
        fig, ax = plt.subplots()
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        self.draw_setup_top(ax)
        plt.savefig(f"{folder}/setuptop.png")
        plt.close()
        fig, ax = plt.subplots()
        plt.plot(self.coincidencias)
        plt.savefig(f"{folder}/coincidencias.png")
        plt.close()

    def evol(self):
        step = self.step
        for evento in self.eventos:
            evento.evol(step)
        self.eventos = [ev for ev in self.eventos if ev.fotones]
        self.check_coincidencias()
        for blanco in self.blancos:
            blanco.check_scattering(step, self.eventos)
        for detector_name, detector, clicks in [
            ("start", self.detector_start, self.clicks_start),
            ("stop", self.detector_stop, self.clicks_stop),
        ]:
            if detector.check_eventos(step, self.eventos):
                clicks.append(-detector.retardo)
                last_detec = detector.ultima_deteccion
                if last_detec is not None:
                    tiempo_entre_clicks = self.time - last_detec
                    detector.estadistica_deteccion.append(tiempo_entre_clicks)
                detector.ultima_deteccion = self.time

    def simular(self, duracion=None):
        if duracion is None:
            duracion = self.duracion
        step = self.step
        prob_emision = self.flujo_eventos * step
        time_print = time()
        time_checkpoint = time()
        while duracion > 0:
            real_time = time()
            if (real_time - time_checkpoint) > self.checkpoint_interval:
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
