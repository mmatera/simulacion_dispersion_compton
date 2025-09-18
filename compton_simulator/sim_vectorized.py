import numpy as np
import numba

PI = np.pi

# =======================
# Vectorized Core Physics
# =======================

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
            u = -1.0 + i * (costheta + 1.0) / 99.0
            total += dof_costheta_numba(u, en)
        return 0.01 * total * (1.0 + costheta)

@numba.njit
def KN_CosTheta_numba(randval, en):
    x0 = 2.0 * randval - 1.0
    rem_steps = 20
    while rem_steps:
        y1 = cof_costheta_numba(x0, en)
        dy = randval - y1
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
    a = 1 - cos_theta ** 2
    w = a / (la + 1.0 / la - a)
    return (1 + w * np.cos(2 * phi)) / (2.0 * PI)

@numba.njit
def cof_phi_numba(phi, cos_theta, en):
    la = 1 / (1 + en * (1 - cos_theta))
    a = 1 - cos_theta ** 2
    w = a / (la + 1.0 / la - a)
    u = 2 * phi
    return 0.5 + 0.25 * (u + w * np.sin(u)) / PI

@numba.njit
def KN_ThetaPhi_numba(randval_theta, randval_phi, en):
    cos_theta = KN_CosTheta_numba(randval_theta, en)
    theta = np.arccos(cos_theta)
    x0 = (2.0 * randval_phi - 1.0) * PI
    rem_steps = 20
    while rem_steps:
        y1 = cof_phi_numba(x0, cos_theta, en)
        dy = randval_phi - y1
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
def norm_numba(v):
    return np.sqrt(np.sum(v ** 2))

@numba.njit
def rotar_numba(vector, eje, angulo):
    doteje = np.dot(vector, eje)
    cross1 = np.cross(eje, np.cross(eje, vector))
    cross2 = np.cross(eje, vector)
    return doteje * eje - np.cos(angulo) * cross1 + np.sin(angulo) * cross2

@numba.njit
def versor_pol_numba(p, pol_v):
    versor = np.array([0.0, 1.0, 0.0]) - p[1] * p / (p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
    norm_versor = norm_numba(versor)
    if norm_versor:
        versor = versor / norm_versor
    else:
        versor = np.array([0.0, 0.0, 1.0])
    if pol_v == 1:
        versor = np.cross(versor, p)
    return versor

@numba.njit
def photon_evol_all(positions, momenta, energies, t):
    n = positions.shape[0]
    for i in range(n):
        if energies[i]:
            positions[i] += t * 29.9 * momenta[i] / energies[i]

@numba.njit
def photon_compton_all(positions, momenta, energies, polarizations, rand_theta, rand_phi):
    n = positions.shape[0]
    for i in range(n):
        p = momenta[i]
        pol = polarizations[i]
        en = norm_numba(p)
        if en == 0.0:
            continue
        p_normed = p / en
        pol_v = pol
        e_pol_in = versor_pol_numba(p_normed, pol_v)
        theta_c, phi_c = KN_ThetaPhi_numba(rand_theta[i], rand_phi[i], en)
        p_out = rotar_numba(p_normed, e_pol_in, theta_c)
        p_out = rotar_numba(p_out, p_normed, phi_c)
        energia = en / (1.0 + en / 511 * (1 - np.cos(theta_c)))
        e_pol_out = versor_pol_numba(p_out, 0)
        prob_V = (np.dot(e_pol_out, e_pol_in)) ** 2
        momenta[i] = p_out * energia
        energies[i] = energia
        polarizations[i] = 0 if np.random.uniform() < prob_V else 1

@numba.njit
def detector_adentro_batch(detector_pos, detector_r, positions):
    mask = np.zeros(positions.shape[0], dtype=np.bool_)
    for i in range(positions.shape[0]):
        pos_rel = positions[i] - detector_pos
        if np.sqrt(np.sum(pos_rel ** 2)) < detector_r:
            mask[i] = True
    return mask

@numba.njit
def photon_aniquilar_batch(positions, momenta, energies, mask):
    for i in range(positions.shape[0]):
        if mask[i]:
            positions[i] = np.array([10000.0, 0.0, 0.0])
            momenta[i] = np.array([0.0, 0.0, 0.0])
            energies[i] = 0.0

@numba.njit
def dispersor_adentro_batch(center, radio, alto, positions):
    mask = np.zeros(positions.shape[0], dtype=np.bool_)
    for i in range(positions.shape[0]):
        pos_rel = center - positions[i]
        if abs(pos_rel[2]) > 0.5 * alto:
            continue
        if pos_rel[0] ** 2 + pos_rel[1] ** 2 > radio ** 2:
            continue
        mask[i] = True
    return mask

@numba.njit
def blanco_compton_batch(positions, momenta, energies, polarizations, mask, rand_theta, rand_phi):
    for i in range(positions.shape[0]):
        if mask[i]:
            p = momenta[i]
            pol = polarizations[i]
            en = norm_numba(p)
            if en == 0.0:
                continue
            p_normed = p / en
            pol_v = pol
            e_pol_in = versor_pol_numba(p_normed, pol_v)
            theta_c, phi_c = KN_ThetaPhi_numba(rand_theta[i], rand_phi[i], en)
            p_out = rotar_numba(p_normed, e_pol_in, theta_c)
            p_out = rotar_numba(p_out, p_normed, phi_c)
            energia = en / (1.0 + en / 511 * (1 - np.cos(theta_c)))
            e_pol_out = versor_pol_numba(p_out, 0)
            prob_V = (np.dot(e_pol_out, e_pol_in)) ** 2
            momenta[i] = p_out * energia
            energies[i] = energia
            polarizations[i] = 0 if np.random.uniform() < prob_V else 1

# ==============================
# OO Interface (original classes)
# ==============================

class Foton:
    def __init__(self, posicion, momento, polarizacion=None, singlete=False):
        self.posicion = np.array(posicion)
        self.momento = np.array(momento)
        self.polarizacion = 0 if polarizacion == "V" else 1 if polarizacion == "H" else np.random.randint(0,2)
        self.energia = float(np.linalg.norm(self.momento))
        self.singlete = singlete

    def __repr__(self):
        pol_str = "V" if self.polarizacion == 0 else "H"
        s = " entangled" if self.singlete else ""
        return f"foton: {self.posicion} pol={pol_str}{s}"

class Evento:
    def __init__(self, CONO_FUENTE, EMITIR_EN_PLANO, HERALDO):
        enpair = 511.0
        costheta_interval = CONO_FUENTE
        arco_azimutal = (-0.00001, 0.00001) if EMITIR_EN_PLANO else (-PI, PI)
        theta1 = np.arccos(np.random.uniform(*costheta_interval))
        phi1 = np.random.uniform(*arco_azimutal)
        p1 = enpair * np.array([
            np.cos(phi1) * np.sin(theta1),
            np.sin(phi1) * np.sin(theta1),
            np.cos(theta1)
        ])
        self.fotones = [
            Foton([0,0,0], p1, singlete=True),
            Foton([0,0,0], -p1, singlete=True)
        ]
        if HERALDO:
            en0 = 1200.0
            theta0 = np.arccos(np.random.uniform(*costheta_interval))
            phi0 = np.random.uniform(*arco_azimutal)
            p0 = en0 * np.array([
                np.cos(phi0) * np.sin(theta0),
                np.sin(phi0) * np.sin(theta0),
                np.cos(theta0)
            ])
            self.fotones.append(Foton([0,0,0], p0, singlete=False))

class Detector:
    def __init__(self, eficiencia, posicion, radio=5.0):
        self.eficiencia = eficiencia
        self.posicion = np.array(posicion)
        self.radio = radio

class Experimento:
    def __init__(self, n_events=10000, CONO_FUENTE=(-0.99999999,1), EMITIR_EN_PLANO=False, HERALDO=True,
                 detector_pos=np.array([10,0,10]), detector_r=2.0, detector_eff=0.9,
                 blanco_center=np.array([0,0,10]), blanco_radio=0.5, blanco_alto=1.0,
                 lambda_dispersion=1.0, lambda_absorcion=1000000.0):
        self.CONO_FUENTE = CONO_FUENTE
        self.EMITIR_EN_PLANO = EMITIR_EN_PLANO
        self.HERALDO = HERALDO
        self.n_events = n_events
        self.eventos = [Evento(CONO_FUENTE, EMITIR_EN_PLANO, HERALDO) for _ in range(n_events)]
        self.detector = Detector(detector_eff, detector_pos, detector_r)
        self.blanco_center = blanco_center
        self.blanco_radio = blanco_radio
        self.blanco_alto = blanco_alto
        self.lambda_dispersion = lambda_dispersion
        self.lambda_absorcion = lambda_absorcion

    def simulate(self, n_steps=1000, t_step=0.001):
        # Gather all photons into arrays
        fotones = []
        for evento in self.eventos:
            fotones.extend(evento.fotones)
        n = len(fotones)
        positions = np.array([f.posicion for f in fotones])
        momenta = np.array([f.momento for f in fotones])
        energies = np.array([f.energia for f in fotones])
        polarizations = np.array([f.polarizacion for f in fotones])
        singlete = np.array([f.singlete for f in fotones])

        for _ in range(n_steps):
            photon_evol_all(positions, momenta, energies, t_step)
            mask = detector_adentro_batch(self.detector.posicion, self.detector.radio, positions)
            n_det = 2 * self.detector.radio / (29.9 * t_step)
            p_det = prob_detection_numba(self.detector.eficiencia, n_det)
            randvals = np.random.uniform(size=n)
            mask_absorbed = mask & (randvals < p_det)
            photon_aniquilar_batch(positions, momenta, energies, mask_absorbed)

            # Blanco interaction
            mask_blanco = dispersor_adentro_batch(self.blanco_center, self.blanco_radio, self.blanco_alto, positions)
            prob_abs = 1.0 - np.exp(-29.9 * t_step / self.lambda_absorcion) if self.lambda_absorcion else 1.0
            prob_disp = 1.0 - np.exp(-29.9 * t_step / self.lambda_dispersion) if self.lambda_dispersion else 1.0
            rand_abs = np.random.uniform(size=n)
            rand_disp = np.random.uniform(size=n)
            mask_abs_blanco = mask_blanco & (rand_abs < prob_abs)
            photon_aniquilar_batch(positions, momenta, energies, mask_abs_blanco)
            mask_compton = mask_blanco & (rand_disp < prob_disp)
            rand_theta = np.random.uniform(size=n)
            rand_phi = np.random.uniform(size=n)
            blanco_compton_batch(positions, momenta, energies, polarizations, mask_compton, rand_theta, rand_phi)

        # Update objects
        for i, f in enumerate(fotones):
            f.posicion = positions[i]
            f.momento = momenta[i]
            f.energia = energies[i]
            f.polarizacion = polarizations[i]
            f.singlete = singlete[i]

    def get_photons(self):
        # For analysis/plotting
        return [f for evento in self.eventos for f in evento.fotones]

# ====================
# Example usage
# ====================

if __name__ == "__main__":
    exp = Experimento()
    exp.simulate(n_steps=1000, t_step=0.001)
    photons = exp.get_photons()
    # Now you can use the OO interface for visualization, analysis, etc.
