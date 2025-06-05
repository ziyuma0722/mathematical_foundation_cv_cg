import os
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import scipy.io


class CreateToolTip(object):
    '''
    create a tooltip for a given widget
    '''

    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.__enter)
        self.widget.bind("<Leave>", self.__close)

    def __enter(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(self.tw, text=self.text, justify='left',
                      bg='yellow', fg='black', relief='solid', borderwidth=1)
        label.pack(ipadx=1)

    def __close(self, event=None):
        if self.tw:
            self.tw.destroy()


class GraphCutGui(Frame):

    def __init__(self, controller, master=None):
        Frame.__init__(self, master)
        self.__controller = controller
        self.__master = master
        self.__drawing = False
        self.__bg_color = 'blue'
        self.__fg_color = 'red'
        self.__button_activated_color = 'gray'
        self.__paint_background = False
        self.__seed_fg = []
        self.__seed_bg = []
        self.__file_dir = os.path.dirname(__file__)
        self.__init_window()

    def __init_window(self):
        self.__master.title("Interactive GraphCut")
        self.__init_canvas()
        self.__init_buttons()
        self.__init_lambda_entry()
        self.__init_brush_size_scale()

    def __init_canvas(self):
        self._normal_image = None
        self._background_image = None
        self._canvas_image = None

        self._canvas = Canvas(self.__master)
        self._canvas.pack(padx=20, pady=20)

        self._canvas.old_coords = None
        self._canvas.bind('<Motion>', self.__mouse_moved)
        self._canvas.bind('<ButtonPress-1>', self.__mouse_pressed)
        self._canvas.bind('<ButtonRelease-1>', self.__mouse_released)

    def __init_buttons(self):

        self._button_icons = []
        icons_directory = self.__file_dir + '/icons/'
        self.__create_button(icons_directory + 'open_file_icon.png', self.__on_file_open_button_clicked, LEFT,
                             "Open Image")

        self.__set_bg_button = self.__create_button(icons_directory + 'open_background_icon.png',
                             self.__on_set_background_image_button_clicked, LEFT,
                             "Set Background Image")

        self.__create_button(icons_directory + 'undo_icon.png', self.__on_reset_image_button_clicked, LEFT,
                             "Reset Image")

        self.__fg_button = self.__create_button(icons_directory + 'fg_icon.png', self.__on_foreground_button_clicked, LEFT, "Foreground")
        self.__orig_button_color = self.__fg_button.cget("background")
        self.__fg_button.config(bg=self.__button_activated_color)

        self.__bg_button = self.__create_button(icons_directory + 'bg_icon.png', self.__on_background_button_clicked, LEFT, "Background")

        self.__create_button(icons_directory + 'scribble.png', self.__on_scribble_button_clicked, LEFT,
                             "Load Scribbles")

        self.__create_button(icons_directory + 'segment_icon.png', self.__on_segmentation_button_clicked, LEFT,
                             "Segment")

    def __init_lambda_entry(self):
        lambda_text = Label(self.__master, text="Lambda")
        lambda_text.pack(side=LEFT)
        self.__lambda = StringVar()
        validation = self.__master.register(self.__only_positive_numbers)
        lambda_entry = Entry(self.__master, textvariable=self.__lambda, validate="key",
                             validatecommand=(validation, '%S'), width=10)
        lambda_entry.pack(side=LEFT, padx=8)
        self.__lambda.set("1.0")

    def __init_brush_size_scale(self):
        brush_text = Label(self.__master, text="Brush Size")
        brush_text.pack(side=LEFT)
        self.__brush_width_scale = Scale(from_=1, to=10, resolution=1, orient=HORIZONTAL)
        self.__brush_width_scale.set(5)
        self.__brush_width_scale.pack(side=LEFT, pady=(1, 15))

    def __only_positive_numbers(self, char):
        return char.isdigit()

    def __create_button(self, button_icon, button_clicked_callback, button_position, button_tool_tip_text):
        self._button_icons.append(PhotoImage(file=button_icon))
        button = Button(self.__master,
                        image=self._button_icons[-1],
                        command=button_clicked_callback)
        button.pack(side=button_position, padx=7)
        CreateToolTip(button, button_tool_tip_text)
        return button

    def __mouse_moved(self, event):
        if self.__drawing:
            x, y = event.x, event.y
            brush_size = float(self.__brush_width_scale.get())
            self.__save_painted_pixels(x, y, brush_size, self._canvas_image.width(), self._canvas_image.height())
            if self._canvas.old_coords:
                x1, y1 = self._canvas.old_coords
                if self.__paint_background:
                    color = self.__bg_color
                else:
                    color = self.__fg_color
                self._canvas.create_line(x, y, x1, y1, width=brush_size, fill=color)
            self._canvas.old_coords = x, y

    def __mouse_pressed(self, event):
        if self._canvas_image is not None:
            self.__drawing = True

    def __mouse_released(self, event):
        self.__drawing = False
        self._canvas.old_coords = None

    def __save_painted_pixels(self, x, y, brush_size, image_width, image_height):
        # get neighbouring pixel indices based on brush size
        brush_range = np.arange(-brush_size, brush_size + 1, 1)
        painted_pixels_i, painted_pixels_j = np.meshgrid(brush_range, brush_range)
        painted_pixels_j = x + painted_pixels_j.flatten()
        painted_pixels_i = y + painted_pixels_i.flatten()
        # remove indices out of bound
        is_not_out_of_bounds = np.invert(
            (painted_pixels_j < 0) + (painted_pixels_j > image_width - 1) + (painted_pixels_i < 0) + (
                    painted_pixels_i > image_height - 1))
        painted_pixels_j = painted_pixels_j[is_not_out_of_bounds].astype(int)
        painted_pixels_i = painted_pixels_i[is_not_out_of_bounds].astype(int)
        new_seeds = [(new_x, new_y) for new_x, new_y in zip(painted_pixels_j, painted_pixels_i)]
        if self.__paint_background:
            self.__seed_bg.extend(new_seeds)
        else:
            self.__seed_fg.extend(new_seeds)

    def __on_foreground_button_clicked(self):
        self.__paint_background = False
        self.__fg_button.config(bg=self.__button_activated_color)
        self.__bg_button.config(bg=self.__orig_button_color)

    def __on_background_button_clicked(self):
        self.__paint_background = True
        self.__fg_button.config(bg=self.__orig_button_color)
        self.__bg_button.config(bg=self.__button_activated_color)

    def __on_scribble_button_clicked(self):
        if self._normal_image is None:
            return

        if self._image_name == 'batman.jpg':
            self.__load_scribbles(self.__file_dir + '/scribbles/scribbles_batman.mat')
            self.__show_scribbles(True)
            self.__show_scribbles(False)
        elif self._image_name == 'VanDamme.jpg':
            self.__load_scribbles(self.__file_dir + '/scribbles/scribbles_JCVD.mat')
            self.__show_scribbles(True)
            self.__show_scribbles(False)

    def __load_scribbles(self, scribbles_dir):
        scribbles = scipy.io.loadmat(scribbles_dir)
        # clear current seeds before loading in the scribbles
        self.__clear_seeds()
        seed_fg = scribbles['seed_fg'] - 1
        seed_bg = scribbles['seed_bg'] - 1
        new_seeds = [(new_x, new_y) for new_x, new_y in zip(seed_fg[:, 0], seed_fg[:, 1])]
        self.__seed_fg.extend(new_seeds)
        new_seeds = [(new_x, new_y) for new_x, new_y in zip(seed_bg[:, 0], seed_bg[:, 1])]
        self.__seed_bg.extend(new_seeds)

    def __show_scribbles(self, show_background):
        if show_background:
            seeds = self.__seed_bg
            color = self.__bg_color
        else:
            seeds = self.__seed_fg
            color = self.__fg_color

        for (x, y) in seeds:
            self._canvas.create_oval(x, y, x, y, width=1, fill=color, outline=color)

    def __on_segmentation_button_clicked(self):
        if self._normal_image is None:
            print("Please choose an image to segment first")
            return
        if len(self.__seed_fg) == 0:
            print("Please choose at least one foreground pixel")
            return
        if len(self.__seed_bg) == 0:
            print("Please choose at least one background pixel")
            return

        lambda_value = float(self.__lambda.get())
        if lambda_value < 0:
            print("Lambda value has to be a positive number")
            return
        self.__controller.segment_image(self._normal_image, self.__seed_fg, self.__seed_bg, lambda_value,
                                        self._background_image)
        # self.__clear_seeds()

    def __on_file_open_button_clicked(self):
        filename = filedialog.askopenfilename(initialdir=self.__file_dir, title="Select file",
                                              filetypes=(
                                                  ("jpeg files", "*.jpg"), ("png files", "*.png"),
                                              ))

        try:
            new_image = Image.open(filename)
            self._image_name = os.path.basename(filename)
            self._normal_image = new_image
            self.set_canvas_image(self._normal_image)
            self.__clear_seeds()
        except (IOError, AttributeError):
            print('Image could not be found. Please try again')

    def __on_set_background_image_button_clicked(self):
        filename = filedialog.askopenfilename(initialdir=self.__file_dir, title="Select file",
                                              filetypes=(
                                                  ("jpeg files", "*.jpg"), ("png files", "*.png"),
                                              ))
        try:
            new_image = Image.open(filename)
            self._background_image = new_image
            self.__set_bg_button.config(bg=self.__button_activated_color)
        except (IOError, AttributeError):
            print('Image could not be found. Please try again')

    def __on_reset_image_button_clicked(self):
        self.__clear_seeds()
        self._background_image = None
        self.__set_bg_button.config(bg=self.__orig_button_color)
        self.set_canvas_image(self._normal_image)

    def __clear_seeds(self):
        self.__seed_fg.clear()
        self.__seed_bg.clear()

    def set_canvas_image(self, image):
        self._canvas.delete("all")
        if image is None:
            return
        self._canvas_image = ImageTk.PhotoImage(image)
        self._canvas.create_image(0, 0, image=self._canvas_image, anchor="nw")
        self._canvas.config(width=self._canvas_image.width(), height=self._canvas_image.height())

    def autoload(self, image_name):
        filename = f'{self.__file_dir}/test_images/{image_name}.jpg'
        new_image = Image.open(filename)
        self._image_name = os.path.basename(filename)
        self._normal_image = new_image
        self.set_canvas_image(self._normal_image)
        self.__clear_seeds()
        self.__on_scribble_button_clicked()
        self.__on_segmentation_button_clicked()
