"""Art Deco design tokens and native Qt primitives for the Studio workspace.

Keep presentation here; analysis settings, image colors, and pixel data belong
to the application and never inherit this palette.
"""
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QDate, QPointF, QRectF, Qt
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QLinearGradient, QPainter,
                           QPalette, QPen, QPolygonF)
from PySide6.QtWidgets import (QFormLayout, QFrame, QGroupBox, QHBoxLayout,
                               QLabel, QScrollArea, QVBoxLayout, QWidget)


@dataclass(frozen=True)
class StudioTokens:
    background: str = '#0A0A0A'
    foreground: str = '#F2F0E4'
    panel: str = '#141414'
    inset: str = '#101010'
    gold: str = '#D4AF37'
    gold_light: str = '#F2E8C4'
    brass: str = '#8E7326'
    border: str = '#554720'
    midnight: str = '#1E3D59'
    muted: str = '#AAA598'
    disabled: str = '#888477'
    disabled_border: str = '#494638'
    focus_background: str = '#191813'
    gold_sheen: str = '#EAD58B'
    gold_dark: str = '#B39134'
    progress_track: str = '#26231B'
    image_background: str = '#030507'
    space: int = 12
    gap: int = 8
    body_px: int = 14
    label_px: int = 11
    title_px: int = 25
    heading_font: str = 'Marcellus'
    body_font: str = 'Josefin Sans'


TOKENS = StudioTokens()
ASSETS = Path(__file__).resolve().parent / 'assets'


def stylesheet():
    t = TOKENS
    check = (ASSETS / 'icons/check.svg').as_posix()
    chevron = (ASSETS / 'icons/chevron.svg').as_posix()
    return f'''
        QWidget {{ color: {t.foreground}; font-family: "{t.body_font}";
                   font-size: {t.body_px}px; }}
        QMainWindow, QDialog {{ background: {t.background}; }}
        QWidget#setupColumn, QWidget#setupFields {{ background: {t.inset}; }}
        QLabel {{ background: transparent; }}
        QLabel[role="brand"] {{ font-family: "{t.heading_font}";
                                 font-size: 28px; color: {t.gold}; }}
        QLabel[role="heading"] {{ font-family: "{t.heading_font}";
                                   font-size: {t.title_px}px; color: {t.foreground}; }}
        QLabel[role="eyebrow"], QLabel[role="fieldLabel"] {{
            font-size: {t.label_px}px; color: {t.muted}; }}
        QLabel[role="hint"], QLabel[role="metadata"] {{ color: {t.muted}; font-size: 12px; }}
        QGroupBox {{ background: {t.panel}; border: 1px solid {t.border};
                     border-radius: 0; margin-top: 24px; padding: 8px 0 0;
                     font-family: "{t.heading_font}"; font-size: 13px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 12px; top: 3px;
                           padding: 0 6px; color: {t.gold};
                           font-family: "{t.heading_font}"; font-size: 13px; }}
        QLineEdit, QComboBox, QDoubleSpinBox {{
            color: {t.foreground}; background: {t.inset};
            border: 0; border-bottom: 1px solid {t.brass}; border-radius: 0;
            min-height: 30px; padding: 6px 7px;
            selection-background-color: {t.midnight}; selection-color: {t.foreground}; }}
        QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {{
            border-bottom: 2px solid {t.gold_light}; background: {t.focus_background}; }}
        QLineEdit:disabled, QComboBox:disabled, QDoubleSpinBox:disabled {{
            color: {t.disabled}; border-bottom-color: {t.disabled_border}; background: {t.inset}; }}
        QLineEdit:read-only {{ color: {t.muted}; }}
        QComboBox {{ padding-right: 25px; }}
        QComboBox::drop-down {{ border: 0; width: 24px; }}
        QComboBox::down-arrow {{ image: url("{chevron}"); width: 10px; height: 8px; }}
        QComboBox QAbstractItemView {{ background: {t.panel}; color: {t.foreground};
            border: 1px solid {t.brass}; selection-background-color: {t.midnight};
            selection-color: {t.foreground}; padding: 4px; }}
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            background: {t.panel}; border: 0; width: 20px; }}
        QPushButton {{ color: {t.gold}; background: transparent; border: 1px solid {t.brass};
            border-radius: 0; min-height: 34px; padding: 6px 10px; }}
        QPushButton:hover {{ background: {t.midnight}; color: {t.gold_light}; border-color: {t.gold}; }}
        QPushButton:pressed {{ background: {t.gold}; color: {t.background}; }}
        QPushButton:focus {{ border: 2px solid {t.gold_light}; }}
        QPushButton[variant="primary"] {{ color: {t.background}; border-color: {t.gold};
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {t.gold_sheen},
                        stop:0.5 {t.gold},stop:1 {t.gold_dark}); }}
        QPushButton[variant="primary"]:hover {{ background: {t.gold_light}; }}
        QPushButton[variant="quiet"] {{ color: {t.muted}; border-color: {t.border}; }}
        QPushButton[variant="quiet"]:hover {{ color: {t.gold_light}; border-color: {t.gold}; }}
        QPushButton[variant="icon"] {{ min-width: 34px; padding: 6px 0; }}
        QPushButton[variant="tool"] {{ min-height: 26px; min-width: 26px; padding: 4px 7px; }}
        QPushButton[variant="color"] {{ color: {t.foreground}; background: {t.panel};
                                        min-height: 26px; padding: 4px 6px; }}
        QPushButton:disabled {{ background: {t.inset}; color: {t.disabled}; border-color: {t.disabled_border}; }}
        QToolButton {{ color: {t.gold}; background: transparent; border: 0;
                      border-bottom: 1px solid {t.border}; border-radius: 0;
                      min-height: 30px; padding: 5px 6px; text-align: left; }}
        QToolButton:hover, QToolButton:checked {{ background: {t.panel}; color: {t.gold_light}; }}
        QToolButton:focus {{ border: 1px solid {t.gold_light}; }}
        QCheckBox {{ spacing: 8px; min-height: 28px; background: transparent; }}
        QCheckBox:focus {{ color: {t.gold_light}; }}
        QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {t.brass};
                               background: {t.inset}; }}
        QCheckBox::indicator:checked {{ background: {t.gold}; border-color: {t.gold};
                                       image: url("{check}"); }}
        QCheckBox::indicator:focus {{ border: 2px solid {t.gold_light}; }}
        QCheckBox:disabled {{ color: {t.disabled}; }}
        QTabBar::tab {{ background: {t.inset}; color: {t.muted}; border: 0;
            border-left: 1px solid {t.border}; border-bottom: 1px solid {t.brass};
            border-radius: 0; min-height: 28px; padding: 8px 12px; }}
        QTabBar::tab:selected {{ background: {t.panel}; color: {t.gold};
                               border-top: 2px solid {t.gold}; }}
        QTabBar::tab:hover {{ color: {t.gold_light}; background: {t.midnight}; }}
        QTabBar:focus {{ border: 1px solid {t.gold_light}; }}
        QTabWidget::pane {{ background: {t.panel}; border: 1px solid {t.border}; }}
        QTabWidget QTabBar::tab {{ min-height: 24px; padding: 6px 10px; font-size: 12px; }}
        QScrollArea {{ border: 0; background: transparent; }}
        QGraphicsView {{ border: 0; background: {t.image_background}; }}
        QTreeWidget, QTableWidget, QPlainTextEdit {{ background: {t.panel}; color: {t.foreground};
            border: 1px solid {t.border}; alternate-background-color: {t.inset};
            selection-background-color: {t.midnight}; selection-color: {t.foreground};
            gridline-color: {t.border}; }}
        QHeaderView::section {{ background: {t.inset}; color: {t.muted};
            border: 0; border-bottom: 1px solid {t.brass}; padding: 8px; font-size: 12px; }}
        QTableWidget::item {{ padding: 6px; }}
        QTreeWidget::item:selected {{ color: {t.foreground}; background: {t.midnight}; }}
        QScrollBar:vertical {{ width: 10px; background: {t.inset}; margin: 0; }}
        QScrollBar::handle:vertical {{ background: {t.brass}; min-height: 28px; margin: 2px; }}
        QScrollBar:horizontal {{ height: 10px; background: {t.inset}; margin: 0; }}
        QScrollBar::handle:horizontal {{ background: {t.brass}; min-width: 28px; margin: 2px; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
        QSplitter::handle {{ background: {t.border}; width: 1px; height: 1px; }}
        QSplitter::handle:hover {{ background: {t.gold}; }}
        QProgressBar {{ border: 0; background: {t.progress_track}; color: {t.foreground};
                       max-height: 20px; text-align: center; }}
        QProgressBar::chunk {{ background: {t.gold}; }}
        QProgressBar[variant="compact"] {{ min-height: 6px; max-height: 6px; }}
        QStatusBar {{ background: {t.inset}; color: {t.muted}; border-top: 1px solid {t.brass};
                      padding: 6px 12px; font-size: 12px; }}
        QStatusBar::item {{ border: 0; }}
        QToolTip {{ background: {t.panel}; color: {t.foreground}; border: 1px solid {t.brass}; padding: 8px; }}
    '''


APP_STYLE = stylesheet()


def apply_theme(app):
    """Use the same native controls and bundled fonts on each desktop platform."""
    if app.property('orionStudioTheme'):
        return
    for name in ('Marcellus-Regular.ttf', 'JosefinSans-Variable.ttf'):
        QFontDatabase.addApplicationFont(str(ASSETS / 'fonts' / name))
    app.setStyle('Fusion')
    palette = QPalette()
    for role, color in {
        QPalette.ColorRole.Window: TOKENS.background,
        QPalette.ColorRole.WindowText: TOKENS.foreground,
        QPalette.ColorRole.Base: TOKENS.inset,
        QPalette.ColorRole.AlternateBase: TOKENS.panel,
        QPalette.ColorRole.Text: TOKENS.foreground,
        QPalette.ColorRole.Button: TOKENS.panel,
        QPalette.ColorRole.ButtonText: TOKENS.gold,
        QPalette.ColorRole.Highlight: TOKENS.midnight,
        QPalette.ColorRole.HighlightedText: TOKENS.foreground,
        QPalette.ColorRole.ToolTipBase: TOKENS.panel,
        QPalette.ColorRole.ToolTipText: TOKENS.foreground,
        QPalette.ColorRole.PlaceholderText: TOKENS.muted,
        QPalette.ColorRole.Link: TOKENS.gold,
    }.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(TOKENS.disabled))
    app.setPalette(palette)
    font = QFont(TOKENS.body_font)
    font.setPixelSize(TOKENS.body_px)
    app.setFont(font)
    app.setStyleSheet(APP_STYLE)
    app.setProperty('orionStudioTheme', True)


def tracked_font(widget, spacing=1.0, uppercase=True):
    font = widget.font()
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    if uppercase:
        font.setCapitalization(QFont.Capitalization.AllUppercase)
    widget.setFont(font)


def text_label(text, role='hint'):
    label = QLabel(text)
    label.setProperty('role', role)
    if role in {'heading', 'eyebrow', 'fieldLabel', 'brand'}:
        tracked_font(label, 1.4 if role in {'heading', 'brand'} else .8,
                     uppercase=role != 'brand')
    if role in {'hint', 'metadata'}:
        label.setWordWrap(True)
    return label


def button_variant(button, variant='default'):
    button.setProperty('variant', variant)
    if variant in {'primary', 'default', 'quiet'}:
        tracked_font(button, .5, uppercase=variant == 'primary')
    return button


def layout_metrics(layout, margin=TOKENS.space, spacing=TOKENS.gap):
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.setSpacing(spacing)
    return layout


class DecoForm(QFormLayout):
    """One label treatment and predictable spacing for static and dynamic forms."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout_metrics(self)
        self.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.setVerticalSpacing(TOKENS.gap)

    def addRow(self, *args):
        if len(args) == 2 and isinstance(args[0], str):
            label = text_label(args[0], 'fieldLabel')
            if isinstance(args[1], QWidget):
                label.setBuddy(args[1])
            args = (label, args[1])
        return super().addRow(*args)


class DecoPanel(QGroupBox):
    def __init__(self, title):
        super().__init__(title.upper())
        self.setAccessibleName(title)
        tracked_font(self, .7)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(QColor(TOKENS.gold), 1))
        top, inset, arm = 27, 5, 12
        painter.drawLine(inset, top + arm, inset, top)
        painter.drawLine(inset, top, inset + arm, top)
        x, y = self.width() - inset - 1, self.height() - inset - 1
        painter.drawLine(x - arm, y, x, y)
        painter.drawLine(x, y, x, y - arm)


class DecoSurface(QWidget):
    def __init__(self, patterned=False):
        super().__init__()
        self.patterned = patterned

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(TOKENS.background))
        if self.patterned:
            color = QColor(TOKENS.gold)
            color.setAlpha(9)
            painter.setPen(color)
            step = TOKENS.space * 2
            for x in range(-self.height(), self.width() + self.height(), step):
                painter.drawLine(x, 0, x + self.height(), self.height())
                painter.drawLine(x, 0, x - self.height(), self.height())


class DecoMark(QWidget):
    """Decorative diamond and orbital geometry; never an interactive control."""
    def __init__(self):
        super().__init__()
        self.setFixedSize(48, 48)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(TOKENS.gold), 1))
        painter.drawPolygon(QPolygonF([QPointF(24, 1), QPointF(47, 24),
                                       QPointF(24, 47), QPointF(1, 24)]))
        painter.translate(24, 24)
        painter.rotate(-35)
        painter.drawEllipse(QRectF(-11, -5, 22, 10))
        painter.rotate(70)
        painter.drawEllipse(QRectF(-11, -5, 22, 10))
        painter.setBrush(QColor(TOKENS.gold))
        painter.drawEllipse(QPointF(0, 0), 2, 2)


class DecoHeader(QWidget):
    def __init__(self):
        super().__init__()
        row = layout_metrics(QHBoxLayout(self), margin=TOKENS.space * 2,
                             spacing=TOKENS.space * 2)
        row.addWidget(DecoMark())
        identity = layout_metrics(QVBoxLayout(), margin=0, spacing=3)
        identity.addWidget(text_label('scikitORION', 'brand'))
        identity.addWidget(text_label('Analysis elevated', 'eyebrow'))
        row.addLayout(identity)
        row.addStretch()
        self.caption = text_label("Analysis date: " + QDate.currentDate().toString('MMMM d, yyyy'), 'eyebrow')
        row.addWidget(self.caption)

    def resizeEvent(self, event):
        self.caption.setVisible(self.width() >= 800)
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, self.height(), self.width() * .6, 0)
        warm = QColor(TOKENS.gold)
        warm.setAlpha(18)
        gradient.setColorAt(0, warm)
        gradient.setColorAt(1, QColor(TOKENS.background))
        painter.fillRect(self.rect(), gradient)
        painter.setPen(QColor(TOKENS.brass))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        painter.setPen(QColor(TOKENS.border))
        painter.drawLine(0, self.height() - 4, self.width(), self.height() - 4)


class DecoImageFrame(QFrame):
    def __init__(self, viewer):
        super().__init__()
        self.setAccessibleName('Image viewing frame')
        self.setMinimumSize(0, 0)
        layout_metrics(QVBoxLayout(self), margin=8, spacing=0).addWidget(viewer)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(TOKENS.background))
        painter.setPen(QColor(TOKENS.brass))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        painter.setPen(QColor(TOKENS.border))
        painter.drawRect(self.rect().adjusted(4, 4, -5, -5))


class DecoInspectionDock(QScrollArea):
    """Prefer a readable inspector height; yield space in smaller windows."""
    def __init__(self, content):
        super().__init__()
        self.setWidgetResizable(True)
        self.setWidget(content)
        self.setMinimumHeight(130)
        self.setMaximumHeight(210)
        self.setAccessibleName('Channel colors and inspection panels')

    def sizeHint(self):
        size = super().sizeHint()
        size.setHeight(200)
        return size
