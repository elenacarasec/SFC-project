import logging

import matplotlib
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow

from data_preparation import Dataset
from model import Network
from ui_form import Ui_MainWindow

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logging.getLogger("matplotlib").setLevel(logging.ERROR)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MplCanvas(FigureCanvas):
    """https://www.pythonguis.com/tutorials/plotting-matplotlib/"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self._connect_buttons()

        self.dataset = Dataset()
        seed = np.random.randint(low=0, high=(2**32 - 1))
        self.nn = self._init_nn(seed=seed)
        self.nn_adam = self._init_nn(optimizer="Adam", seed=seed)

        self.labelNumEpochsTrained.setText(f"Epochs trained: {self.nn.current_epoch}")
        self.update_test_loss()

        # Create a Matplotlib canvas
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.matplotlibBox.addWidget(self.canvas)
        self.plot_graph()

    def plot_graph(self):
        self.update_plot()
        self.show()

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QtCore.QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        self.xdata = list(range(1, len(self.nn.train_cost) + 1))
        self.ydata1 = self.nn.train_cost
        self.ydata2 = self.nn_adam.train_cost

        self.canvas.axes.cla()  # Clear the canvas.
        self.canvas.axes.plot(self.xdata, self.ydata1, "g", label="Backpropagation")
        self.canvas.axes.plot(
            self.xdata, self.ydata2, "b", label="Backpropagation + Adam"
        )

        self.canvas.axes.set_xlabel("Epochs")
        self.canvas.axes.set_ylabel("MSE Loss")
        self.canvas.axes.legend()

        # Adjust limits and aspect ratio for a tight layout
        right_xlim = 2 if len(self.xdata) < 2 else len(self.xdata)
        self.canvas.axes.set_xlim(left=1, right=right_xlim)
        self.canvas.axes.set_ylim(ymin=0)
        self.canvas.axes.set_aspect(
            "auto"
        )
        self.canvas.fig.tight_layout()

        # Trigger the canvas to update and redraw.
        self.canvas.draw()

    def _connect_buttons(self):
        self.updateModelButton.clicked.connect(self.update_button_clicked)
        self.buttonStepBack.clicked.connect(self.step_back_button_clicked)
        self.buttonStepIn.clicked.connect(self.step_in_button_clicked)
        self.buttonRun.clicked.connect(self.run_button_clicked)
        self.buttonReset.clicked.connect(self.reset_button_clicked)

    def _init_nn(self, optimizer=None, seed=0):
        # Read values from the input fields
        self.epochs = self.spinBox.value()
        alpha = self.doubleSpinBox.value()
        beta1 = self.doubleSpinBox_2.value()
        beta2 = self.doubleSpinBox_3.value()

        return Network(
            [self.dataset.X_train.shape[0], 64, 64, 1],
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            optimizer=optimizer,
            seed=seed,
        )

    def update_button_clicked(self):
        seed = np.random.randint(low=0, high=(2**32 - 1))
        self.nn = self._init_nn(seed=seed)
        self.nn_adam = self._init_nn(optimizer="Adam", seed=seed)

        self.labelNumEpochsTrained.setText(f"Epochs trained: {self.nn.current_epoch}")
        self.update_test_loss()

    def step_back_button_clicked(self):
        self.nn.step_back()
        self.nn_adam.step_back()
        self.labelNumEpochsTrained.setText(f"Epochs trained: {self.nn.current_epoch}")
        self.update_test_loss()

    def step_in_button_clicked(self):
        self.nn.train_epoch(self.dataset.X_train, self.dataset.y_train)
        self.nn_adam.train_epoch(self.dataset.X_train, self.dataset.y_train)

        self.nn.current_epoch += 1
        self.nn_adam.current_epoch += 1
        self.labelNumEpochsTrained.setText(f"Epochs trained: {self.nn.current_epoch}")

        self.update_test_loss()


    def run_button_clicked(self):
        logger.debug("Run clicked")
        self.nn.train(self.dataset.X_train, self.dataset.y_train, self.epochs)
        self.nn_adam.train(self.dataset.X_train, self.dataset.y_train, self.epochs)
        self.labelNumEpochsTrained.setText(f"Epochs trained: {self.nn.current_epoch}")

        self.update_test_loss()

        logger.debug("Model trained")

    def update_test_loss(self):
        cost = self.nn.test(self.dataset.X_test, self.dataset.y_test)
        self.labelBPLossValue.setText(f"{round(cost, 5)}")
        cost = self.nn_adam.test(self.dataset.X_test, self.dataset.y_test)
        self.labelBPAdamLossValue.setText(f"{round(cost, 5)}")

    def reset_button_clicked(self):
        seed = np.random.randint(low=0, high=(2**32 - 1))
        self.nn = self._init_nn(seed=seed)
        self.nn_adam = self._init_nn(optimizer="Adam", seed=seed)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
