
import csv
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
G = 0.2


def make_disk_galaxy(
    n,
    center,
    bulk_velocity,
    radius,
    thickness,
    central_mass,
    body_mass=1.0,
    rotation_direction=1.0,
    speed_scale=1.0,
    angle_offset=0.0,
    seed=SEED
):
    rng = np.random.default_rng(seed)

    center = np.asarray(center, dtype=np.float64)
    bulk_velocity = np.asarray(bulk_velocity, dtype=np.float64)

    particle_count = n - 1

    # Distribuzione uniforme per area, più densa verso il centro.
    radial_factor = np.sqrt(rng.random(particle_count))
    radii = radius * radial_factor

    angles = rng.uniform(0.0, 2.0 * np.pi, particle_count)
    angles += angle_offset

    # Rumore radiale per evitare una distribuzione troppo regolare.
    radii += rng.normal(
        0.0,
        thickness * (0.25 + 0.75 * radial_factor),
        particle_count
    )
    radii = np.maximum(radii, 0.15)

    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)

    # Velocità orbitale approssimata:
    # v = sqrt(G * M(r) / r)
    enclosed_mass = (
        central_mass
        + body_mass * particle_count * radial_factor**2
    )

    orbital_speed = speed_scale * np.sqrt(
        G * enclosed_mass / np.maximum(radii, 0.5)
    )

    # Piccola dispersione per rendere la galassia meno artificiale.
    orbital_speed *= rng.normal(1.0, 0.04, particle_count)

    tangent_x = -np.sin(angles) * rotation_direction
    tangent_y = np.cos(angles) * rotation_direction

    vx = bulk_velocity[0] + tangent_x * orbital_speed
    vy = bulk_velocity[1] + tangent_y * orbital_speed

    masses = np.full(particle_count, body_mass, dtype=np.float64)

    # Corpo centrale.
    x = np.append(x, center[0])
    y = np.append(y, center[1])
    vx = np.append(vx, bulk_velocity[0])
    vy = np.append(vy, bulk_velocity[1])
    masses = np.append(masses, central_mass)

    return x, y, vx, vy, masses


def make_ring(
    n,
    center,
    radius,
    width,
    central_mass,
    body_mass,
    bulk_velocity=(0.0, 0.0),
    speed_scale=1.0,
    seed=SEED
):
    rng = np.random.default_rng(seed)

    center = np.asarray(center, dtype=np.float64)
    bulk_velocity = np.asarray(bulk_velocity, dtype=np.float64)

    angles = rng.uniform(0.0, 2.0 * np.pi, n)
    radii = rng.normal(radius, width, n)
    radii = np.maximum(radii, 0.5)

    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)

    speed = speed_scale * np.sqrt(
        G * central_mass / np.maximum(radii, 0.5)
    )
    speed *= rng.normal(1.0, 0.025, n)

    vx = bulk_velocity[0] - np.sin(angles) * speed
    vy = bulk_velocity[1] + np.cos(angles) * speed

    masses = np.full(n, body_mass, dtype=np.float64)

    return x, y, vx, vy, masses


def combine_systems(*systems):
    return tuple(
        np.concatenate([system[field] for system in systems])
        for field in range(5)
    )


def center_momentum(vx, vy, masses):
    total_mass = np.sum(masses)

    mean_vx = np.sum(vx * masses) / total_mass
    mean_vy = np.sum(vy * masses) / total_mass

    return vx - mean_vx, vy - mean_vy


def write_galaxy(name, x, y, vx, vy, masses, round_to=7):
    with open(
        f"./galaxies/{name}.csv",
        "w",
        newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)

        for values in zip(x, y, vx, vy, masses):
            writer.writerow([
                round(float(values[0]), round_to),
                round(float(values[1]), round_to),
                round(float(values[2]), round_to),
                round(float(values[3]), round_to),
                round(float(values[4]), round_to)
            ])


def plot_galaxy(x, y, vx, vy, masses):
    plt.figure(figsize=(12, 10))

    display_size = np.clip(np.sqrt(masses), 0.3, 90.0)

    plt.scatter(
        x,
        y,
        s=display_size,
        alpha=0.55
    )

    # Disegna solo una parte delle velocità per non saturare il grafico.
    stride = max(1, len(x) // 2500)

    plt.quiver(
        x[::stride],
        y[::stride],
        vx[::stride],
        vy[::stride],
        alpha=0.25,
        scale=900
    )

    plt.axis("equal")
    plt.title(f"Triple galaxy collision — {len(x):,} bodies")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


# ============================================================
# SISTEMA "GALACTIC CHAOS"
# 180.000 particelle
# Ideale per confrontare Naive vs Barnes-Hut
# ============================================================

# ---------------- Galassia principale ----------------

main = make_disk_galaxy(
    n=70000,
    center=(0.0, 0.0),
    bulk_velocity=(0.0, 0.0),
    radius=65,
    thickness=4,
    central_mass=55000,
    body_mass=1.0,
    rotation_direction=1,
    speed_scale=0.84,
    seed=1
)

# ---------------- Satellite sinistro ----------------

left = make_disk_galaxy(
    n=22000,
    center=(-260.0, -120.0),
    bulk_velocity=(8.5, 4.0),
    radius=26,
    thickness=3,
    central_mass=12000,
    body_mass=0.8,
    rotation_direction=-1,
    speed_scale=0.88,
    angle_offset=0.5,
    seed=2
)

# ---------------- Satellite superiore ----------------

top = make_disk_galaxy(
    n=22000,
    center=(230.0, 180.0),
    bulk_velocity=(-6.8, -5.8),
    radius=24,
    thickness=3,
    central_mass=11000,
    body_mass=0.8,
    rotation_direction=1,
    speed_scale=0.86,
    angle_offset=-0.3,
    seed=3
)

# ---------------- Tre anelli ----------------

ring1 = make_ring(
    n=12000,
    center=(0.0, 0.0),
    radius=95,
    width=2.5,
    central_mass=70000,
    body_mass=0.40,
    speed_scale=0.82,
    seed=4
)

ring2 = make_ring(
    n=9000,
    center=(0.0, 0.0),
    radius=145,
    width=4.0,
    central_mass=70000,
    body_mass=0.25,
    speed_scale=0.76,
    seed=5
)

ring3 = make_ring(
    n=7000,
    center=(0.0, 0.0),
    radius=220,
    width=8.0,
    central_mass=70000,
    body_mass=0.18,
    speed_scale=0.69,
    seed=6
)

# ---------------- Ammassi globulari ----------------

cluster1 = make_disk_galaxy(
    n=6000,
    center=(-170.0, 220.0),
    bulk_velocity=(3.0, -2.0),
    radius=8,
    thickness=1,
    central_mass=2200,
    body_mass=0.5,
    rotation_direction=1,
    speed_scale=0.95,
    seed=11
)

cluster2 = make_disk_galaxy(
    n=6000,
    center=(180.0, -210.0),
    bulk_velocity=(-2.5, 2.8),
    radius=9,
    thickness=1,
    central_mass=2200,
    body_mass=0.5,
    rotation_direction=-1,
    speed_scale=0.95,
    seed=12
)

cluster3 = make_disk_galaxy(
    n=5000,
    center=(280.0, 40.0),
    bulk_velocity=(-4.0, 1.0),
    radius=7,
    thickness=0.8,
    central_mass=1800,
    body_mass=0.5,
    rotation_direction=1,
    speed_scale=0.93,
    seed=13
)

cluster4 = make_disk_galaxy(
    n=5000,
    center=(-300.0, -40.0),
    bulk_velocity=(4.0, -1.0),
    radius=7,
    thickness=0.8,
    central_mass=1800,
    body_mass=0.5,
    rotation_direction=-1,
    speed_scale=0.93,
    seed=14
)

# ---------------- Halo diffuso ----------------

def make_halo(
    n,
    radius,
    center=(0.0, 0.0),
    mass=0.12,
    seed=30
):
    rng = np.random.default_rng(seed)

    theta = rng.uniform(0.0, 2*np.pi, n)
    r = radius * np.sqrt(rng.random(n))
    r *= (1.3 + rng.random(n))

    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    # piccole velocità casuali
    vx = rng.normal(0.0, 0.15, n)
    vy = rng.normal(0.0, 0.15, n)

    masses = np.full(n, mass)

    return x, y, vx, vy, masses


halo = make_halo(
    n=16000,
    radius=420,
    mass=0.12,
    seed=30
)

# ---------------- Combina tutto ----------------

x, y, vx, vy, masses = combine_systems(
    main,
    left,
    top,
    ring1,
    ring2,
    ring3,
    cluster1,
    cluster2,
    cluster3,
    cluster4,
    halo
)

vx, vy = center_momentum(vx, vy, masses)

print(f"Corpi totali: {len(x):,}")
print(f"Massa totale: {np.sum(masses):,.2f}")

write_galaxy(
    "galactic_chaos_180k",
    x,
    y,
    vx,
    vy,
    masses
)

plot_galaxy(
    x,
    y,
    vx,
    vy,
    masses
)