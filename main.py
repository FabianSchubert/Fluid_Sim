from ui import App

from simulator import Simulator

if __name__ == "__main__":
    sim = Simulator((700, 700), backend="GPU")

    app = App(sim, size=(700,700))
    app.run()
