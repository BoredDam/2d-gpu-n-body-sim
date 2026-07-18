import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_bounding_boxes(csv_path):
    fig, ax = plt.subplots()

    with open(csv_path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue

            try:
                x = float(row[0])
                y = float(row[1])
                radius = float(row[2])
            except (ValueError, IndexError):
                print(f"Riga {row_number} non valida: {row}")
                continue

            bounding_box = Rectangle(
                (x - radius, y - radius),
                2 * radius,
                2 * radius,
                linewidth=0.7,
                fill=False
            )

            ax.add_patch(bounding_box)

            # Centro della bounding box.
            #ax.scatter(x, y, s=1)

    ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Bounding boxes del quadtree")
    ax.grid(True)

    return plt


plt = plot_bounding_boxes("./outputs/sim4/sim4_frame_0.csv")

plt.savefig("./images/bounding-box4.pdf")
plt.show()