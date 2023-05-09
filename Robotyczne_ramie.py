import chess
import chess.engine
import chess.svg
import serial
import keyboard
import time
import cv2
from cairosvg import svg2png
import numpy as np
import pygame
import io


pp = 1

def nothing(X):
    pass

def thresold_calibreation(img):
    cv2.namedWindow("thresold_calibration")
    cv2.createTrackbar("thresold", "thresold_calibration", 0, 255, nothing)
    while True:
        t =  cv2.getTrackbarPos("thresold", "thresold_calibration")
        matrix,thresold = cv2.threshold(img,t,255,cv2.THRESH_BINARY)
        cv2.imshow("thresold",thresold)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return t     

#Niwelowanie zniekształcen rybiego oka
def disortion(path, nr):
    # Wczytanie obrazka
    img = cv2.imread(path)

    # Współczynniki do korekcji efektu rybiego oka DŁUGI NAPIS ABY SZYBKO ZOBACZYC GDZIE W KODZIE SIE TO ZNAJDUJE KALIBRACJA
    k1 = -0.07
    k2 = 1

    # Środek obrazka
    x0 = img.shape[1] / 2
    y0 = img.shape[0] / 2

    # Promień obrazka
    r = min(x0, y0)

    # Tworzenie macierzy przekształcenia
    map_x = np.zeros_like(img[:,:,0]).astype(np.float32)
    map_y = np.zeros_like(img[:,:,0]).astype(np.float32)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            dx = x - x0
            dy = y - y0
            distance = (dx ** 2 + dy ** 2) ** 0.5
            if distance == 0:
                new_x = x
                new_y = y
            else:
                theta = (distance / r) ** k1 * k2
                new_x = x0 + theta * dx
                new_y = y0 + theta * dy
            map_x[y,x] = new_x
            map_y[y,x] = new_y

    # Korekcja efektu rybiego oka
    corrected_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    # Zapisanie nowego obrazka
    cv2.imwrite(f'snapshot_{nr}.jpg', corrected_img)


def polozenie(path):
    # Wczytanie obrazu
    img = cv2.imread(path)

    # Konwersja do skali szarości
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binaryzacja
    thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

    # Wyszukiwanie konturów
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sortowanie konturów według pola powierzchni (od największego do najmniejszego)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Wybór tylko 64 największych konturów (kwadratów)
    contours = contours[:64]

    # Rysowanie linii na obrazie
    h, w = img.shape[:2]
    sq_h, sq_w = h // 8, w // 8

    for i in range(8):
        for j in range(8):
            x1, y1 = j * sq_w, i * sq_h
            x2, y2 = (j + 1) * sq_w, (i + 1) * sq_h
            
            # Rysowanie linii pionowych
            cv2.line(img, (x1, y1), (x1, y2), (0, 255, 0), 2)
            
            # Rysowanie linii poziomych
            cv2.line(img, (x1, y1), (x2, y1), (0, 255, 0), 2)
            
            # Wycinanie obszaru odpowiadającego danemu kwadratowi
            square = thresh[y1:y2, x1:x2]
            
            # Wyszukiwanie konturu białego obiektu w kwadracie
            square_contours, _ = cv2.findContours(square, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            square_contours = sorted(square_contours, key=cv2.contourArea, reverse=True)
            if square_contours:
                # Znalezienie środka ciężkości konturu białego obiektu
                M = cv2.moments(square_contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + x1
                    cy = int(M["m01"] / M["m00"]) + y1
                    
                    # Wypisanie położenia białego obiektu na obrazie
                    cv2.putText(img, f'{cx}, {cy}', (x1 + 5, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    print("pozycja x" , cx)
                    print("pozycja y" , cy)

    # Wyświetlenie obrazu
    cv2.imshow('image', img)
    cv2.waitKey(0)


def change_size(ww):
   #Ustawienie kamery i dopasowanie jej krawedzi
    img = cv2.imread(f'snapshot_{ww}.jpg')
    #zmiana wspolrzednych od lewego gornego rogu DŁUGI NAPIS ABY SZYBKO ZOBACZYC GDZIE W KODZIE SIE TO ZNAJDUJE KALIBRACJA
    input_points = np.float32([[457,132],[496,454],[137,132],[107,451]])
    width = 600
    height = 600
    converated_points = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(input_points, converated_points)
    img_output = cv2.warpPerspective(img,matrix,(width,height))
    cv2.imwrite(f'snapshot_{ww}.jpg', img_output)


def znajdowanie_pola():
    # Wczytanie obrazu o wymiarach 600x600
    img = cv2.imread('tresh.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Podzielenie obrazu na 64 kwadraty
    rows, cols = gray.shape
    sq_size = rows // 8
    for i in range(8):
        for j in range(8):
            x = i * sq_size
            y = j * sq_size
            if (i + j) % 2 == 0:
                cv2.rectangle(img, (x, y), (x + sq_size, y + sq_size), (0, 0, 0), -1)
            else:
                cv2.rectangle(img, (x, y), (x + sq_size, y + sq_size), (255, 255, 255), -1)

    # Znalezienie dwóch kwadratów z największą ilością białego pola
    white_areas = []
    for i in range(8):
        for j in range(8):
            x = i * sq_size
            y = j * sq_size
            roi = gray[y:y+sq_size, x:x+sq_size]
            white_areas.append((cv2.countNonZero(roi), (i, j)))

    white_areas.sort(reverse=True)
    top_two = white_areas[:2]
    top_squares = []
    for area in top_two:
        x, y = area[1]
        top_squares.append((x, y))
        
    # Zaznaczenie wybranych kwadratów na obrazie
    i = 0
    for square in top_squares:
        x = square[0] * sq_size
        y = square[1] * sq_size
        cv2.rectangle(img, (x, y), (x + sq_size, y + sq_size), (0, 255, 0), 2)
        # print("Square with highest white count: ", convert_to_chess_notation(x, y))
        
        if i == 0:
            pierwsze = convert_to_chess_notation(x, y)
            i = i + 1
        if i == 1:
            drugie = convert_to_chess_notation(x, y)    

    razem = pierwsze+drugie

    return razem
    # Wyświetlenie obrazu z zaznaczonymi kwadratami
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convert_to_chess_notation(x, y):
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    row = 8 - (y // 100)
    col = letters[x // 100]
    return col + str(row)


def wyswietlanie():
    # Wczytanie obrazu szachownicy SVG i konwersja na format PNG
    board_svg = chess.svg.board(board)
    board_png = svg2png(board_svg)

    # Konwersja obrazu PNG na obiekt Pygame
    board_surface = pygame.image.load(io.BytesIO(board_png))

    # Ustawienie pozycji szachownicy na ekranie
    board_rect = board_surface.get_rect(center=screen.get_rect().center)

    screen.blit(board_surface, board_rect)

    # Odświeżenie ekranu
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()


def display_board(board):
    # Wyświetl planszę w konsoli
    print(board)


def display_move(move):
    global x
    global y 
    # Podziel ruch na dwie części
    x = str(move)[:2]
    y = str(move)[2:]


def get_player_move(board):
    invalid = False
    while True:
        time.sleep(3)    

        # Wprowadzenie ruch
        if(invalid==True):
            print("Cofnij ruch trzymając pionka w dłoni i nacisnij s")
            keyboard.wait('s')
            _, frame = capture.read()
            cv2.imwrite('snapshot_1.jpg', frame)
            change_size(1)
            path = (f'snapshot_1.jpg')
            disortion(path,1)
            print("Puść cofnietego pionka i nacisnij s")
            keyboard.wait('s')
            _, frame = capture.read()
            cv2.imwrite('snapshot_1.jpg', frame)
            change_size(1)
            path = (f'snapshot_1.jpg')
            disortion(path,1)

        else:
            _, frame = capture.read()
            cv2.imwrite('snapshot_1.jpg', frame)
            change_size(1)
            path = (f'snapshot_1.jpg')
            disortion(path,1)

        print('Wykonaj ruch i nacisnij s')
        keyboard.wait('s')
        _, frame = capture.read()
        cv2.imwrite('snapshot_2.jpg', frame)
        change_size(2)
        path = (f'snapshot_2.jpg')
        disortion(path,2)

        image1 = cv2.imread(f'snapshot_1.jpg')
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

        image2 = cv2.imread(f'snapshot_2.jpg')
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(image1,image2)
        diff = cv2.resize(diff,(800,800))

        diff_gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
        cv2.imwrite("Difference_GrayScale_image.jpg",diff_gray)


        #KALIBRACJA!!!!
        # value = thresold_calibreation(diff_gray)
        # matrix,thresold = cv2.threshold(diff_gray,value,255,cv2.THRESH_BINARY)
        # cv2.imshow("thresold",thresold)


        ret, thresh = cv2.threshold(diff_gray, 54, 255, cv2.THRESH_BINARY)          
        cv2.imwrite("tresh.jpg",thresh)
        sciezka = ('tresh.jpg')
        posuniecie = znajdowanie_pola()
    
        wyswietlanie()
        _, frame = capture.read()
        move_input = posuniecie 
        try:
            move = chess.Move.from_uci(move_input)
            if move in board.legal_moves:
                invalid=False
                return move
            else:
                move_input = posuniecie[2:] + posuniecie[:2]
                move = chess.Move.from_uci(move_input)
                if move in board.legal_moves:
                    invalid=False
                    return move

                invalid=True
        except ValueError:
            print("Niepoprawny format ruchu! Spróbuj ponownie.")
            invalid=True

# Ustawienia połączenia szeregowego
ser = serial.Serial('COM5', 115200)

invalid = False
print ("Hello szachy")
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

print('Czekaj..')
cv2.waitKey(0)
time.sleep(2)
_, frame = capture.read()
path1 = (f'test_1.jpg')
time.sleep(1)

_, frame = capture.read()
path1 = (f'test_1.jpg')
cv2.imwrite(path1, frame)
cv2.waitKey(0)

path = (f'snapshot_1.jpg')
cv2.imwrite(path, frame)

img = cv2.imread(path)

change_size(1)
disortion(path,pp)
cv2.imshow("oryginal", frame)

# Utwórz nową grę
board = chess.Board()

# Utwórz silnik Stockfish
engine = chess.engine.SimpleEngine.popen_uci("stockfish-windows-2022-x86-64-avx2.exe")
pygame.init()

# Ustawienie wymiarów okna
WINDOW_SIZE = (400, 400)
screen = pygame.display.set_mode(WINDOW_SIZE)
wyswietlanie()
_, frame = capture.read()

while not board.is_game_over():
    wyswietlanie()
    cv2.imshow("oryginal", frame)
    
    _, frame = capture.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Wyświetl planszę
    display_board(board)
    zbicie =False
    # Poproś gracza o ruch lub wykonaj ruch za pomocą silnika Stockfish
    if board.turn == chess.WHITE:
        move = get_player_move(board)
    else:
        result = engine.play(board, chess.engine.Limit(time=2.0))
        move = result.move
        display_move(move)
        if board.is_capture(move):
            # print("Stockfish zbił figure")
            zbicie = True
        else:
            # print("Stockfish nie zbił figury")
            zbicie = False
            
    wyswietlanie()
    _, frame = capture.read()
    if(board.turn == chess.BLACK):
        print("Ruch Robota czekaj")
        if(zbicie==True):
            if(y=='0'):
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='a1'):
                ser.write(('G1 X-11.5 Y32.5 Z33 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-11.5 Y27 Z36.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y23 Z39.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-11.5 Y27 Z36.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y32.5 Z33 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='a2'):
                ser.write(('G1 X-12 Y28.5 Z27.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-12 Y23 Z31 F1000\n').encode())
                ser.write(('G1 X-12 Y18.5 Z36 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-12 Y23 Z31 F1000\n').encode())
                ser.write(('G1 X-12 Y28.5 Z27.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='a3'):
                ser.write(('G1 X-13.5 Y22.5 Z24 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-13 Y16 Z29 F1000\n').encode())
                ser.write(('G1 X-13 Y13 Z32.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-13 Y16 Z29 F1000\n').encode())
                ser.write(('G1 X-13.5 Y22.5 Z24 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='a4'):
                ser.write(('G1 X-14.5 Y20.5 Z19 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-14 Y12.5 Z26 F1000\n').encode())
                ser.write(('G1 X-14 Y8.5 Z30.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-14 Y12.5 Z26 F1000\n').encode())
                ser.write(('G1 X-14.5 Y20.5 Z19 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='a5'):
                ser.write(('G1 X-15.5 Y17 Z15.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-15.5 Y9.5 Z22.5 F1000\n').encode())
                ser.write(('G1 X-15.5 Y4.5 Z28.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(3)
                ser.write(('G1 X-15.5 Y9.5 Z22.5 F1000\n').encode())
                ser.write(('G1 X-15.5 Y17 Z15.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='a6'):
                ser.write(('G1 X-17 Y13.5 Z12.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-17 Y5.5 Z20 F1000\n').encode())
                ser.write(('G1 X-17 Y0 Z26.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-17 Y5.5 Z20 F1000\n').encode())
                ser.write(('G1 X-17 Y13.5 Z12.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='a7'):
                ser.write(('G1 X-19 Y11.5 Z9 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-19 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X-18.5 Y-3 Z24 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-19 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X-19 Y11.5 Z9 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='a8'):
                ser.write(('G1 X-22 Y6.5 Z8 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-21.5 Y0 Z14.5 F1000\n').encode())
                ser.write(('G1 X-21.5 Y-7 Z22.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-21.5 Y0 Z14.5 F1000\n').encode())
                ser.write(('G1 X-22 Y6.5 Z8 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='b1'):
                ser.write(('G1 X-9 Y29.5 Z31.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-8.5 Y24.5 Z36 F1000\n').encode())
                ser.write(('G1 X-8.5 Y22 Z39.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-8.5 Y24.5 Z36 F1000\n').encode())
                ser.write(('G1 X-9 Y29.5 Z31.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='b2'):
                ser.write(('G1 X-9 Y26 Z26.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-9 Y20.5 Z30.5 F1000\n').encode())
                ser.write(('G1 X-9 Y16.5 Z35 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-9 Y20.5 Z30.5 F1000\n').encode())
                ser.write(('G1 X-9 Y26 Z26.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='b3'):
                ser.write(('G1 X-9.5 Y22 Z21.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-9.5 Y16 Z26.5 F1000\n').encode())
                ser.write(('G1 X-9.5 Y12 Z31.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-9.5 Y16 Z26.5 F1000\n').encode())
                ser.write(('G1 X-9.5 Y22 Z21.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='b4'):
                ser.write(('G1 X-10.5 Y19 Z18 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-10.5 Y11.5 Z24.5 F1000\n').encode())
                ser.write(('G1 X-10.5 Y7.5 Z29.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-10.5 Y11.5 Z24.5 F1000\n').encode())
                ser.write(('G1 X-10.5 Y19 Z18 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='b5'):
                ser.write(('G1 X-11.5 Y17 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-11.5 Y8.5 Z20.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y3 Z27 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-11.5 Y8.5 Z20.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y17 Z14 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='b6'):
                ser.write(('G1 X-13 Y15 Z10 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-12.5 Y5 Z18 F1000\n').encode())
                ser.write(('G1 X-12.5 Y-1.5 Z25 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-12.5 Y5 Z18 F1000\n').encode())
                ser.write(('G1 X-13 Y15 Z10 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='b7'):
                ser.write(('G1 X-14.5 Y11.5 Z7 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-14 Y2 Z15.5 F1000\n').encode())
                ser.write(('G1 X-14 Y-5.5 Z23 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-14 Y2 Z15.5 F1000\n').encode())
                ser.write(('G1 X-14.5 Y11.5 Z7 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='b8'):
                ser.write(('G1 X-16.5 Y6.5 Z5.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-16.5 Y-1.5 Z13 F1000\n').encode())
                ser.write(('G1 X-16.5 Y-10 Z22.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(2)
                ser.write(('G1 X-16.5 Y-1.5 Z13 F1000\n').encode())
                ser.write(('G1 X-16.5 Y6.5 Z5.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='c1'):
                ser.write(('G1 X-5.5 Y30 Z30 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-5 Y24 Z34.5 F1000\n').encode())
                ser.write(('G1 X-5.5 Y21 Z38 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-5 Y24 Z34.5 F1000\n').encode())
                ser.write(('G1 X-5.5 Y30 Z30 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='c2'):
                ser.write(('G1 X-6 Y25.5 Z25 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-6 Y19.5 Z30 F1000\n').encode())
                ser.write(('G1 X-6 Y15.5 Z34 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(5)
                ser.write(('G1 X-6 Y19.5 Z30 F1000\n').encode())
                ser.write(('G1 X-6 Y25.5 Z25 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='c3'):
                ser.write(('G1 X-6 Y21 Z21 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-6 Y14.5 Z26.5 F1000\n').encode())
                ser.write(('G1 X-6 Y10.5 Z31 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(5)
                ser.write(('G1 X-6 Y14.5 Z26.5 F1000\n').encode())
                ser.write(('G1 X-6 Y21 Z21 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='c4'):
                ser.write(('G1 X-6.5 Y18 Z17 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-6.5 Y10.5 Z23.5 F1000\n').encode())
                ser.write(('G1 X-6.5 Y6 Z29 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-6.5 Y10.5 Z23.5 F1000\n').encode())
                ser.write(('G1 X-6.5 Y18 Z17 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='c5'):
                ser.write(('G1 X-7 Y14.5 Z13.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-7 Y7 Z20 F1000\n').encode())
                ser.write(('G1 X-7 Y2 Z26.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-7 Y7 Z20 F1000\n').encode())
                ser.write(('G1 X-7 Y14.5 Z13.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='c6'):
                ser.write(('G1 X-8.5 Y14 Z8.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-8 Y3.5 Z17.5 F1000\n').encode())
                ser.write(('G1 X-8 Y-2.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-8 Y3.5 Z17.5 F1000\n').encode())
                ser.write(('G1 X-8.5 Y14 Z8.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='c7'):
                ser.write(('G1 X-9.5 Y8.5 Z6.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-9 Y-0.5 Z15 F1000\n').encode())
                ser.write(('G1 X-9 Y-7 Z22.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-9 Y-0.5 Z15 F1000\n').encode())
                ser.write(('G1 X-9.5 Y8.5 Z6.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='c8'):
                ser.write(('G1 X-10 Y5 Z4.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-10 Y-3.5 Z12 F1000\n').encode())
                ser.write(('G1 X-10.5 Y-12.5 Z22 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-10 Y-3.5 Z12 F1000\n').encode())
                ser.write(('G1 X-10 Y5 Z4.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='d1'):
                ser.write(('G1 X-3 Y28.5 Z30 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-2.5 Y23 Z34 F1000\n').encode())
                ser.write(('G1 X-2.5 Y20.5 Z37 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y23 Z34 F1000\n').encode())
                ser.write(('G1 X-3 Y28.5 Z30 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='d2'):
                ser.write(('G1 X-2.5 Y24.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-2.5 Y19 Z29 F1000\n').encode())
                ser.write(('G1 X-2.5 Y15 Z33.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y19 Z29 F1000\n').encode())
                ser.write(('G1 X-2.5 Y24.5 Z24.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='d3'):
                ser.write(('G1 X-2.5 Y21 Z20 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-2.5 Y14 Z26 F1000\n').encode())
                ser.write(('G1 X-2.5 Y10 Z30.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y14 Z26 F1000\n').encode())
                ser.write(('G1 X-2.5 Y21 Z20 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='d4'):
                ser.write(('G1 X-2.5 Y17.5 Z16.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-2.5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X-3 Y6 Z28.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X-2.5 Y17.5 Z16.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='d5'):
                ser.write(('G1 X-3 Y14.5 Z13 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-2.5 Y7 Z19.5 F1000\n').encode())
                ser.write(('G1 X-3 Y1.5 Z26 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-2.5 Y7 Z19.5 F1000\n').encode())
                ser.write(('G1 X-3 Y14.5 Z13 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='d6'):
                ser.write(('G1 X-3.5 Y11.5 Z9.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-3 Y3 Z17 F1000\n').encode())
                ser.write(('G1 X-3 Y-3 Z24 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-3 Y3 Z17 F1000\n').encode())
                ser.write(('G1 X-3.5 Y11.5 Z9.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='d7'):
                ser.write(('G1 X-4 Y7 Z6.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-3.5 Y-0.5 Z13.5 F1000\n').encode())
                ser.write(('G1 X-3.5 Y-8 Z22 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-3.5 Y-0.5 Z13.5 F1000\n').encode())
                ser.write(('G1 X-4 Y7 Z6.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='d8'):
                ser.write(('G1 X-4.5 Y6 Z3 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-4 Y-7 Z15 F1000\n').encode())
                ser.write(('G1 X-4 Y-13.5 Z22 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-4 Y-7 Z15 F1000\n').encode())
                ser.write(('G1 X-4.5 Y6 Z3 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='e1'):
                ser.write(('G1 X0.5 Y28.5 Z29.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X0.5 Y23 Z35 F1000\n').encode())
                ser.write(('G1 X0.5 Y20.5 Z37 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X0.5 Y23 Z35 F1000\n').encode())
                ser.write(('G1 X0.5 Y28.5 Z29.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='e2'):
                ser.write(('G1 X0.5 Y24.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X0.5 Y19 Z28.5 F1000\n').encode())
                ser.write(('G1 X0.5 Y15 Z33.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X0.5 Y19 Z28.5 F1000\n').encode())
                ser.write(('G1 X0.5 Y24.5 Z24.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='e3'):
                ser.write(('G1 X1 Y21 Z20 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X1 Y14 Z25.5 F1000\n').encode())
                ser.write(('G1 X1 Y10 Z30 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X1 Y14 Z25.5 F1000\n').encode())
                ser.write(('G1 X1 Y21 Z20 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='e4'):
                ser.write(('G1 X1 Y17.5 Z16.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X1.5 Y10 Z23 F1000\n').encode())
                ser.write(('G1 X1.5 Y5.5 Z28 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X1.5 Y10 Z23 F1000\n').encode())
                ser.write(('G1 X1 Y17.5 Z16.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='e5'):
                ser.write(('G1 X1 Y15 Z12 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X1.5 Y7 Z19 F1000\n').encode())
                ser.write(('G1 X1.5 Y1 Z26 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X1.5 Y7 Z19 F1000\n').encode())
                ser.write(('G1 X1 Y15 Z12 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='e6'):
                ser.write(('G1 X1.5 Y11 Z9.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X2 Y3 Z16.5 F1000\n').encode())
                ser.write(('G1 X2 Y-3 Z24 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X2 Y3 Z16.5 F1000\n').encode())
                ser.write(('G1 X1.5 Y11 Z9.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='e7'):
                ser.write(('G1 X2 Y7 Z6.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X2 Y0 Z13 F1000\n').encode())
                ser.write(('G1 X2 Y-8 Z22 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X2 Y0 Z13 F1000\n').encode())
                ser.write(('G1 X2 Y7 Z6.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='e8'):
                ser.write(('G1 X2.5 Y4.5 Z3.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X3 Y-6 Z13 F1000\n').encode())
                ser.write(('G1 X3 Y-14 Z21.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X3 Y-6 Z13 F1000\n').encode())
                ser.write(('G1 X2.5 Y4.5 Z3.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='f1'):
                ser.write(('G1 X3.5 Y29 Z30 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X3.5 Y23.5 Z34 F1000\n').encode())
                ser.write(('G1 X3.5 Y20.5 Z37.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X3.5 Y23.5 Z34 F1000\n').encode())
                ser.write(('G1 X3.5 Y29 Z30 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='f2'):
                ser.write(('G1 X4 Y25 Z25 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X4 Y19 Z29.5 F1000\n').encode())
                ser.write(('G1 X4 Y15.5 Z33.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X4 Y19 Z29.5 F1000\n').encode())
                ser.write(('G1 X4 Y25 Z25 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='f3'):
                ser.write(('G1 X4.5 Y21.5 Z20 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X4.5 Y14.5 Z25.5 F1000\n').encode())
                ser.write(('G1 X4.5 Y11 Z30 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X4.5 Y14.5 Z25.5 F1000\n').encode())
                ser.write(('G1 X4.5 Y21.5 Z20 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='f4'):
                ser.write(('G1 X5 Y19 Z16 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X5 Y6 Z28 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X5 Y19 Z16 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='f5'):
                ser.write(('G1 X5.5 Y15.5 Z12.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X6 Y7.5 Z19 F1000\n').encode())
                ser.write(('G1 X6 Y1.5 Z26 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X6 Y7.5 Z19 F1000\n').encode())
                ser.write(('G1 X5.5 Y15.5 Z12.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='f6'):
                ser.write(('G1 X6.5 Y11.5 Z9.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X7 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X7 Y-2.5 Z24 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X7 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X6.5 Y11.5 Z9.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='f7'):
                ser.write(('G1 X7.5 Y6 Z8 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X8 Y-0.5 Z14 F1000\n').encode())
                ser.write(('G1 X8 Y-7.5 Z22 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X8 Y-0.5 Z14 F1000\n').encode())
                ser.write(('G1 X7.5 Y6 Z8 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='f8'):
                ser.write(('G1 X8.5 Y7.5 Z2 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X9 Y-4 Z12 F1000\n').encode())
                ser.write(('G1 X9 Y-12.5 Z21 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X9 Y-4 Z12 F1000\n').encode())
                ser.write(('G1 X8.5 Y7.5 Z2 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(10)
            elif(y=='g1'):
                ser.write(('G1 X6 Y30 Z31.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X6 Y25 Z35 F1000\n').encode())
                ser.write(('G1 X6 Y21.5 Z37.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X6 Y25 Z35 F1000\n').encode())
                ser.write(('G1 X6 Y30 Z31.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='g2'):
                ser.write(('G1 X7 Y25 Z25.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X7 Y20 Z29.5 F1000\n').encode())
                ser.write(('G1 X7 Y16.5 Z34 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X7 Y20 Z29.5 F1000\n').encode())
                ser.write(('G1 X7 Y25 Z25.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='g3'):
                ser.write(('G1 X8 Y21.5 Z21 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X8 Y15 Z26.5 F1000\n').encode())
                ser.write(('G1 X8 Y11.5 Z30.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X8 Y15 Z26.5 F1000\n').encode())
                ser.write(('G1 X8 Y21.5 Z21 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='g4'):
                ser.write(('G1 X9 Y18.5 Z17 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X9 Y11.5 Z23 F1000\n').encode())
                ser.write(('G1 X9 Y6.5 Z28.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X9 Y11.5 Z23 F1000\n').encode())
                ser.write(('G1 X9 Y18.5 Z17 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='g5'):
                ser.write(('G1 X9.5 Y15.5 Z13.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X10 Y8 Z20 F1000\n').encode())
                ser.write(('G1 X10 Y2.5 Z26 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X10 Y8 Z20 F1000\n').encode())
                ser.write(('G1 X9.5 Y15.5 Z13.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='g6'):
                ser.write(('G1 X11.5 Y12 Z10.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X11.5 Y4 Z17.5 F1000\n').encode())
                ser.write(('G1 X11.5 Y-1.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X11.5 Y4 Z17.5 F1000\n').encode())
                ser.write(('G1 X11.5 Y12 Z10.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='g7'):
                ser.write(('G1 X13 Y7.5 Z8 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X13 Y0.5 Z14.5 F1000\n').encode())
                ser.write(('G1 X13 Y-6.5 Z22.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X13 Y0.5 Z14.5 F1000\n').encode())
                ser.write(('G1 X13 Y7.5 Z8 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='g8'):
                ser.write(('G1 X16 Y6 Z4.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X16 Y-2 Z12 F1000\n').encode())
                ser.write(('G1 X16 Y-8.5 Z19 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X16 Y-2 Z12 F1000\n').encode())
                ser.write(('G1 X16 Y6 Z4.5 F1000\n').encode())
                time.sleep(10)
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X16 Y6 Z4.5 F1000\n').encode())
                ser.write(('G1 X16 Y-2 Z12 F1000\n').encode())
                ser.write(('G1 X16 Y-8.5 Z19 F1000\n').encode())
                ser.write(('M3 S45\n').encode())
                ser.write(('G1 X16 Y-2 Z12 F1000\n').encode())
                ser.write(('G1 X16 Y6 Z4.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='h1'):
                ser.write(('G1 X9 Y30.5 Z32 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X9 Y25.5 Z35.5 F1000\n').encode())
                ser.write(('G1 X9 Y23 Z39 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X9 Y25.5 Z35.5 F1000\n').encode())
                ser.write(('G1 X9 Y30.5 Z32 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='h2'):
                ser.write(('G1 X10 Y26 Z26.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X10 Y21 Z30.5 F1000\n').encode())
                ser.write(('G1 X10 Y17.5 Z35 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X10 Y21 Z30.5 F1000\n').encode())
                ser.write(('G1 X10 Y26 Z26.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='h3'):
                ser.write(('G1 X11 Y21.5 Z22.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X11 Y17 Z26 F1000\n').encode())
                ser.write(('G1 X11.5 Y13 Z31 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X11 Y17 Z26 F1000\n').encode())
                ser.write(('G1 X11 Y21.5 Z22.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='h4'):
                ser.write(('G1 X12.5 Y19.5 Z17.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X12.5 Y13 Z23 F1000\n').encode())
                ser.write(('G1 X12.5 Y8.5 Z29 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X12.5 Y13 Z23 F1000\n').encode())
                ser.write(('G1 X12.5 Y19.5 Z17.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='h5'):
                ser.write(('G1 X13.5 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X14 Y9 Z21 F1000\n').encode())
                ser.write(('G1 X14 Y4 Z27 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X14 Y9 Z21 F1000\n').encode())
                ser.write(('G1 X13.5 Y17.5 Z14 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='h6'):
                ser.write(('G1 X15.5 Y14 Z11 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X16 Y6 Z18 F1000\n').encode())
                ser.write(('G1 X16 Y0.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X16 Y6 Z18 F1000\n').encode())
                ser.write(('G1 X15.5 Y14 Z11 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='h7'):
                ser.write(('G1 X17.5 Y9.5 Z8.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X18 Y2 Z16 F1000\n').encode())
                ser.write(('G1 X18 Y-4 Z22.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X18 Y2 Z16 F1000\n').encode())
                ser.write(('G1 X17.5 Y9.5 Z8.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
            elif(y=='h8'):
                ser.write(('G1 X21 Y9.5 Z3.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X21 Y1.5 Z11.5 F1000\n').encode())
                ser.write(('G1 X21.5 Y-6.5 Z20 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(10)
                ser.write(('G1 X21 Y1.5 Z11.5 F1000\n').encode())
                ser.write(('G1 X21 Y9.5 Z3.5 F1000\n').encode())

                time.sleep(5)
                ser.write(('G1 X25 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie

            time.sleep(2)
            if(x=='0'):
                ser.write(('G1 y0 Y0 Z0 F1000\n').encode())
            elif(x=='a1'):
                ser.write(('G1 X-11.5 Y32.5 Z33 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-11.5 Y27 Z36.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y23 Z39.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-11.5 Y27 Z36.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y32.5 Z33 F1000\n').encode())
            elif(x=='a2'):
                ser.write(('G1 X-12 Y28.5 Z27.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-12 Y23 Z31 F1000\n').encode())
                ser.write(('G1 X-12 Y18.5 Z36 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-12 Y23 Z31 F1000\n').encode())
                ser.write(('G1 X-12 Y28.5 Z27.5 F1000\n').encode())
            elif(x=='a3'):
                ser.write(('G1 X-13.5 Y22.5 Z24 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-13 Y16 Z29 F1000\n').encode())
                ser.write(('G1 X-13 Y13 Z32.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-13 Y16 Z29 F1000\n').encode())
                ser.write(('G1 X-13.5 Y22.5 Z24 F1000\n').encode())
            elif(x=='a4'):
                ser.write(('G1 X-14.5 Y20.5 Z19 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-14 Y12.5 Z26 F1000\n').encode())
                ser.write(('G1 X-14 Y8.5 Z30.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-14 Y12.5 Z26 F1000\n').encode())
                ser.write(('G1 X-14.5 Y20.5 Z19 F1000\n').encode())
            elif(x=='a5'):
                ser.write(('G1 X-15.5 Y17 Z15.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-15.5 Y9.5 Z22.5 F1000\n').encode())
                ser.write(('G1 X-15.5 Y4.5 Z28.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(3)
                ser.write(('G1 X-15.5 Y9.5 Z22.5 F1000\n').encode())
                ser.write(('G1 X-15.5 Y17 Z15.5 F1000\n').encode())
            elif(x=='a6'):
                ser.write(('G1 X-17 Y13.5 Z12.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-17 Y5.5 Z20 F1000\n').encode())
                ser.write(('G1 X-17 Y0 Z26.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-17 Y5.5 Z20 F1000\n').encode())
                ser.write(('G1 X-17 Y13.5 Z12.5 F1000\n').encode())
            elif(x=='a7'):
                ser.write(('G1 X-19 Y11.5 Z9 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-19 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X-18.5 Y-3 Z24 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-19 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X-19 Y11.5 Z9 F1000\n').encode())
            elif(x=='a8'):
                ser.write(('G1 X-22 Y6.5 Z8 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-21.5 Y0 Z14.5 F1000\n').encode())
                ser.write(('G1 X-21.5 Y-7 Z22.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-21.5 Y0 Z14.5 F1000\n').encode())
                ser.write(('G1 X-22 Y6.5 Z8 F1000\n').encode())
            elif(x=='b1'):
                ser.write(('G1 X-9 Y29.5 Z31.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-8.5 Y24.5 Z36 F1000\n').encode())
                ser.write(('G1 X-8.5 Y22 Z39.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-8.5 Y24.5 Z36 F1000\n').encode())
                ser.write(('G1 X-9 Y29.5 Z31.5 F1000\n').encode())
            elif(x=='b2'):
                ser.write(('G1 X-9 Y26 Z26.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-9 Y20.5 Z30.5 F1000\n').encode())
                ser.write(('G1 X-9 Y16.5 Z35 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-9 Y20.5 Z30.5 F1000\n').encode())
                ser.write(('G1 X-9 Y26 Z26.5 F1000\n').encode())
            elif(x=='b3'):
                ser.write(('G1 X-9.5 Y22 Z21.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-9.5 Y16 Z26.5 F1000\n').encode())
                ser.write(('G1 X-9.5 Y12 Z31.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-9.5 Y16 Z26.5 F1000\n').encode())
                ser.write(('G1 X-9.5 Y22 Z21.5 F1000\n').encode())
            elif(x=='b4'):
                ser.write(('G1 X-10.5 Y19 Z18 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-10.5 Y11.5 Z24.5 F1000\n').encode())
                ser.write(('G1 X-10.5 Y7.5 Z29.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-10.5 Y11.5 Z24.5 F1000\n').encode())
                ser.write(('G1 X-10.5 Y19 Z18 F1000\n').encode())
            elif(x=='b5'):
                ser.write(('G1 X-11.5 Y17 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-11.5 Y8.5 Z20.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y3 Z27 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-11.5 Y8.5 Z20.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y17 Z14 F1000\n').encode())
            elif(x=='b6'):
                ser.write(('G1 X-13 Y15 Z10 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-12.5 Y5 Z18 F1000\n').encode())
                ser.write(('G1 X-12.5 Y-1.5 Z25 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-12.5 Y5 Z18 F1000\n').encode())
                ser.write(('G1 X-13 Y15 Z10 F1000\n').encode())
            elif(x=='b7'):
                ser.write(('G1 X-14.5 Y11.5 Z7 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-14 Y2 Z15.5 F1000\n').encode())
                ser.write(('G1 X-14 Y-5.5 Z23 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-14 Y2 Z15.5 F1000\n').encode())
                ser.write(('G1 X-14.5 Y11.5 Z7 F1000\n').encode())
            elif(x=='b8'):
                ser.write(('G1 X-16.5 Y6.5 Z5.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-16.5 Y-1.5 Z13 F1000\n').encode())
                ser.write(('G1 X-16.5 Y-10 Z22.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-16.5 Y-1.5 Z13 F1000\n').encode())
                ser.write(('G1 X-16.5 Y6.5 Z5.5 F1000\n').encode())
            elif(x=='c1'):
                ser.write(('G1 X-5.5 Y30 Z30 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-5 Y24 Z34.5 F1000\n').encode())
                ser.write(('G1 X-5.5 Y21 Z38 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-5 Y24 Z34.5 F1000\n').encode())
                ser.write(('G1 X-5.5 Y30 Z30 F1000\n').encode())
            elif(x=='c2'):
                ser.write(('G1 X-6 Y25.5 Z25 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-6 Y19.5 Z30 F1000\n').encode())
                ser.write(('G1 X-6 Y15.5 Z34 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-6 Y19.5 Z30 F1000\n').encode())
                ser.write(('G1 X-6 Y25.5 Z25 F1000\n').encode())
            elif(x=='c3'):
                ser.write(('G1 X-6 Y21 Z21 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-6 Y14.5 Z26.5 F1000\n').encode())
                ser.write(('G1 X-6 Y10.5 Z31 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-6 Y14.5 Z26.5 F1000\n').encode())
                ser.write(('G1 X-6 Y21 Z21 F1000\n').encode())
            elif(x=='c4'):
                ser.write(('G1 X-6.5 Y18 Z17 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-6.5 Y10.5 Z23.5 F1000\n').encode())
                ser.write(('G1 X-6.5 Y6 Z29 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-6.5 Y10.5 Z23.5 F1000\n').encode())
                ser.write(('G1 X-6.5 Y18 Z17 F1000\n').encode())
            elif(x=='c5'):
                ser.write(('G1 X-7 Y14.5 Z13.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-7 Y7 Z20 F1000\n').encode())
                ser.write(('G1 X-7 Y2 Z26.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-7 Y7 Z20 F1000\n').encode())
                ser.write(('G1 X-7 Y14.5 Z13.5 F1000\n').encode())
            elif(x=='c6'):
                ser.write(('G1 X-8.5 Y14 Z8.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-8 Y3.5 Z17.5 F1000\n').encode())
                ser.write(('G1 X-8 Y-2.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-8 Y3.5 Z17.5 F1000\n').encode())
                ser.write(('G1 X-8.5 Y14 Z8.5 F1000\n').encode())
            elif(x=='c7'):
                ser.write(('G1 X-9.5 Y8.5 Z6.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-9 Y-0.5 Z15 F1000\n').encode())
                ser.write(('G1 X-9 Y-7 Z22.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-9 Y-0.5 Z15 F1000\n').encode())
                ser.write(('G1 X-9.5 Y8.5 Z6.5 F1000\n').encode())
            elif(x=='c8'):
                ser.write(('G1 X-10 Y5 Z4.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-10 Y-3.5 Z12 F1000\n').encode())
                ser.write(('G1 X-10.5 Y-12.5 Z22 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-10 Y-3.5 Z12 F1000\n').encode())
                ser.write(('G1 X-10 Y5 Z4.5 F1000\n').encode())
            elif(x=='d1'):
                ser.write(('G1 X-3 Y28.5 Z30 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-2.5 Y23 Z34 F1000\n').encode())
                ser.write(('G1 X-2.5 Y20.5 Z37 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y23 Z34 F1000\n').encode())
                ser.write(('G1 X-3 Y28.5 Z30 F1000\n').encode())
            elif(x=='d2'):
                ser.write(('G1 X-2.5 Y24.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-2.5 Y19 Z29 F1000\n').encode())
                ser.write(('G1 X-2.5 Y15 Z33.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y19 Z29 F1000\n').encode())
                ser.write(('G1 X-2.5 Y24.5 Z24.5 F1000\n').encode())
            elif(x=='d3'):
                ser.write(('G1 X-2.5 Y21 Z20 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-2.5 Y14 Z26 F1000\n').encode())
                ser.write(('G1 X-2.5 Y10 Z30.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y14 Z26 F1000\n').encode())
                ser.write(('G1 X-2.5 Y21 Z20 F1000\n').encode())
            elif(x=='d4'):
                ser.write(('G1 X-2.5 Y17.5 Z16.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-2.5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X-3 Y6 Z28.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X-2.5 Y17.5 Z16.5 F1000\n').encode())
            elif(x=='d5'):
                ser.write(('G1 X-3 Y14.5 Z13 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-2.5 Y7 Z19.5 F1000\n').encode())
                ser.write(('G1 X-3 Y1.5 Z26 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(3)
                ser.write(('G1 X-2.5 Y7 Z19.5 F1000\n').encode())
                ser.write(('G1 X-3 Y14.5 Z13 F1000\n').encode())
            elif(x=='d6'):
                ser.write(('G1 X-3.5 Y11.5 Z9.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-3 Y3 Z17 F1000\n').encode())
                ser.write(('G1 X-3 Y-3 Z24 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-3 Y3 Z17 F1000\n').encode())
                ser.write(('G1 X-3.5 Y11.5 Z9.5 F1000\n').encode())
            elif(x=='d7'):
                ser.write(('G1 X-4 Y7 Z6.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-3.5 Y-0.5 Z13.5 F1000\n').encode())
                ser.write(('G1 X-3.5 Y-8 Z22 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-3.5 Y-0.5 Z13.5 F1000\n').encode())
                ser.write(('G1 X-4 Y7 Z6.5 F1000\n').encode())
            elif(x=='d8'):
                ser.write(('G1 X-4.5 Y6 Z3 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-4 Y-7 Z15 F1000\n').encode())
                ser.write(('G1 X-4 Y-13.5 Z22 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-4 Y-7 Z15 F1000\n').encode())
                ser.write(('G1 X-4.5 Y6 Z3 F1000\n').encode())
            elif(x=='e1'):
                ser.write(('G1 X0.5 Y28.5 Z29.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X0.5 Y23 Z35 F1000\n').encode())
                ser.write(('G1 X0.5 Y20.5 Z37 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X0.5 Y23 Z35 F1000\n').encode())
                ser.write(('G1 X0.5 Y28.5 Z29.5 F1000\n').encode())
            elif(x=='e2'):
                ser.write(('G1 X0.5 Y24.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X0.5 Y19 Z28.5 F1000\n').encode())
                ser.write(('G1 X0.5 Y15 Z33.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X0.5 Y19 Z28.5 F1000\n').encode())
                ser.write(('G1 X0.5 Y24.5 Z24.5 F1000\n').encode())
            elif(x=='e3'):
                ser.write(('G1 X1 Y21 Z20 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X1 Y14 Z25.5 F1000\n').encode())
                ser.write(('G1 X1 Y10 Z30 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X1 Y14 Z25.5 F1000\n').encode())
                ser.write(('G1 X1 Y21 Z20 F1000\n').encode())
            elif(x=='e4'):
                ser.write(('G1 X1 Y17.5 Z16.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X1.5 Y10 Z23 F1000\n').encode())
                ser.write(('G1 X1.5 Y5.5 Z28 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X1.5 Y10 Z23 F1000\n').encode())
                ser.write(('G1 X1 Y17.5 Z16.5 F1000\n').encode())
            elif(x=='e5'):
                ser.write(('G1 X1 Y15 Z12 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X1.5 Y7 Z19 F1000\n').encode())
                ser.write(('G1 X1.5 Y1 Z26 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X1.5 Y7 Z19 F1000\n').encode())
                ser.write(('G1 X1 Y15 Z12 F1000\n').encode())
            elif(x=='e6'):
                ser.write(('G1 X1.5 Y11 Z9.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X2 Y3 Z16.5 F1000\n').encode())
                ser.write(('G1 X2 Y-3 Z24 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X2 Y3 Z16.5 F1000\n').encode())
                ser.write(('G1 X1.5 Y11 Z9.5 F1000\n').encode())
            elif(x=='e7'):
                ser.write(('G1 X2 Y7 Z6.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X2 Y0 Z13 F1000\n').encode())
                ser.write(('G1 X2 Y-8 Z22 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X2 Y0 Z13 F1000\n').encode())
                ser.write(('G1 X2 Y7 Z6.5 F1000\n').encode())
            elif(x=='e8'):
                ser.write(('G1 X2.5 Y4.5 Z3.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X3 Y-6 Z13 F1000\n').encode())
                ser.write(('G1 X3 Y-14 Z21.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X3 Y-6 Z13 F1000\n').encode())
                ser.write(('G1 X2.5 Y4.5 Z3.5 F1000\n').encode())
            elif(x=='f1'):
                ser.write(('G1 X3.5 Y29 Z30 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X3.5 Y23.5 Z34 F1000\n').encode())
                ser.write(('G1 X3.5 Y20.5 Z37.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X3.5 Y23.5 Z34 F1000\n').encode())
                ser.write(('G1 X3.5 Y29 Z30 F1000\n').encode())
            elif(x=='f2'):
                ser.write(('G1 X4 Y25 Z25 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X4 Y19 Z29.5 F1000\n').encode())
                ser.write(('G1 X4 Y15.5 Z33.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X4 Y19 Z29.5 F1000\n').encode())
                ser.write(('G1 X4 Y25 Z25 F1000\n').encode())
            elif(x=='f3'):
                ser.write(('G1 X4.5 Y21.5 Z20 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X4.5 Y14.5 Z25.5 F1000\n').encode())
                ser.write(('G1 X4.5 Y11 Z30 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X4.5 Y14.5 Z25.5 F1000\n').encode())
                ser.write(('G1 X4.5 Y21.5 Z20 F1000\n').encode())
            elif(x=='f4'):
                ser.write(('G1 X5 Y19 Z16 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X5 Y6 Z28 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X5 Y19 Z16 F1000\n').encode())
            elif(x=='f5'):
                ser.write(('G1 X5.5 Y15.5 Z12.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X6 Y7.5 Z19 F1000\n').encode())
                ser.write(('G1 X6 Y1.5 Z26 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X6 Y7.5 Z19 F1000\n').encode())
                ser.write(('G1 X5.5 Y15.5 Z12.5 F1000\n').encode())
            elif(x=='f6'):
                ser.write(('G1 X6.5 Y11.5 Z9.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X7 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X7 Y-2.5 Z24 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X7 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X6.5 Y11.5 Z9.5 F1000\n').encode())
            elif(x=='f7'):
                ser.write(('G1 X7.5 Y6 Z8 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X8 Y-0.5 Z14 F1000\n').encode())
                ser.write(('G1 X8 Y-7.5 Z22 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X8 Y-0.5 Z14 F1000\n').encode())
                ser.write(('G1 X7.5 Y6 Z8 F1000\n').encode())
            elif(x=='f8'):
                ser.write(('G1 X8.5 Y7.5 Z2 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X9 Y-4 Z12 F1000\n').encode())
                ser.write(('G1 X9 Y-12.5 Z21 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X9 Y-4 Z12 F1000\n').encode())
                ser.write(('G1 X8.5 Y7.5 Z2 F1000\n').encode())
                time.sleep(10)
            elif(x=='g1'):
                ser.write(('G1 X6 Y30 Z31.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X6 Y25 Z35 F1000\n').encode())
                ser.write(('G1 X6 Y21.5 Z37.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X6 Y25 Z35 F1000\n').encode())
                ser.write(('G1 X6 Y30 Z31.5 F1000\n').encode())
            elif(x=='g2'):
                ser.write(('G1 X7 Y25 Z25.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X7 Y20 Z29.5 F1000\n').encode())
                ser.write(('G1 X7 Y16.5 Z34 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X7 Y20 Z29.5 F1000\n').encode())
                ser.write(('G1 X7 Y25 Z25.5 F1000\n').encode())
            elif(x=='g3'):
                ser.write(('G1 X8 Y21.5 Z21 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X8 Y15 Z26.5 F1000\n').encode())
                ser.write(('G1 X8 Y11.5 Z30.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X8 Y15 Z26.5 F1000\n').encode())
                ser.write(('G1 X8 Y21.5 Z21 F1000\n').encode())
            elif(x=='g4'):
                ser.write(('G1 X9 Y18.5 Z17 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X9 Y11.5 Z23 F1000\n').encode())
                ser.write(('G1 X9 Y6.5 Z28.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X9 Y11.5 Z23 F1000\n').encode())
                ser.write(('G1 X9 Y18.5 Z17 F1000\n').encode())
            elif(x=='g5'):
                ser.write(('G1 X9.5 Y15.5 Z13.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X10 Y8 Z20 F1000\n').encode())
                ser.write(('G1 X10 Y2.5 Z26 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X10 Y8 Z20 F1000\n').encode())
                ser.write(('G1 X9.5 Y15.5 Z13.5 F1000\n').encode())
            elif(x=='g6'):
                ser.write(('G1 X11.5 Y12 Z10.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X11.5 Y4 Z17.5 F1000\n').encode())
                ser.write(('G1 X11.5 Y-1.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X11.5 Y4 Z17.5 F1000\n').encode())
                ser.write(('G1 X11.5 Y12 Z10.5 F1000\n').encode())
            elif(x=='g7'):
                ser.write(('G1 X13 Y7.5 Z8 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X13 Y0.5 Z14.5 F1000\n').encode())
                ser.write(('G1 X13 Y-6.5 Z22.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X13 Y0.5 Z14.5 F1000\n').encode())
                ser.write(('G1 X13 Y7.5 Z8 F1000\n').encode())
            elif(x=='g8'):
                ser.write(('G1 X16 Y6 Z4.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X16 Y-2 Z12 F1000\n').encode())
                ser.write(('G1 X16 Y-8.5 Z19 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X16 Y-2 Z12 F1000\n').encode())
                ser.write(('G1 X16 Y6 Z4.5 F1000\n').encode())
                time.sleep(10)
                ser.write(('G1 X16 Y6 Z4.5 F1000\n').encode())
                ser.write(('G1 X16 Y-2 Z12 F1000\n').encode())
                ser.write(('G1 X16 Y-8.5 Z19 F1000\n').encode())
                ser.write(('M3 S45\n').encode())
                ser.write(('G1 X16 Y-2 Z12 F1000\n').encode())
                ser.write(('G1 X16 Y6 Z4.5 F1000\n').encode())
            elif(x=='h1'):
                ser.write(('G1 X9 Y30.5 Z32 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X9 Y25.5 Z35.5 F1000\n').encode())
                ser.write(('G1 X9 Y23 Z39 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X9 Y25.5 Z35.5 F1000\n').encode())
                ser.write(('G1 X9 Y30.5 Z32 F1000\n').encode())
            elif(x=='h2'):
                ser.write(('G1 X10 Y26 Z26.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X10 Y21 Z30.5 F1000\n').encode())
                ser.write(('G1 X10 Y17.5 Z35 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X10 Y21 Z30.5 F1000\n').encode())
                ser.write(('G1 X10 Y26 Z26.5 F1000\n').encode())
            elif(x=='h3'):
                ser.write(('G1 X11 Y21.5 Z22.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X11 Y17 Z26 F1000\n').encode())
                ser.write(('G1 X11.5 Y13 Z31 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X11 Y17 Z26 F1000\n').encode())
                ser.write(('G1 X11 Y21.5 Z22.5 F1000\n').encode())
            elif(x=='h4'):
                ser.write(('G1 X12.5 Y19.5 Z17.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X12.5 Y13 Z23 F1000\n').encode())
                ser.write(('G1 X12.5 Y8.5 Z29 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X12.5 Y13 Z23 F1000\n').encode())
                ser.write(('G1 X12.5 Y19.5 Z17.5 F1000\n').encode())
            elif(x=='h5'):
                ser.write(('G1 X13.5 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X14 Y9 Z21 F1000\n').encode())
                ser.write(('G1 X14 Y4 Z27 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X14 Y9 Z21 F1000\n').encode())
                ser.write(('G1 X13.5 Y17.5 Z14 F1000\n').encode())
            elif(x=='h6'):
                ser.write(('G1 X15.5 Y14 Z11 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X16 Y6 Z18 F1000\n').encode())
                ser.write(('G1 X16 Y0.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X16 Y6 Z18 F1000\n').encode())
                ser.write(('G1 X15.5 Y14 Z11 F1000\n').encode())
            elif(x=='h7'):
                ser.write(('G1 X17.5 Y9.5 Z8.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X18 Y2 Z16 F1000\n').encode())
                ser.write(('G1 X18 Y-4 Z22.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X18 Y2 Z16 F1000\n').encode())
                ser.write(('G1 X17.5 Y9.5 Z8.5 F1000\n').encode())
            elif(x=='h8'):
                ser.write(('G1 X21 Y9.5 Z3.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X21 Y1.5 Z11.5 F1000\n').encode())
                ser.write(('G1 X21.5 Y-6.5 Z20 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X21 Y1.5 Z11.5 F1000\n').encode())
                ser.write(('G1 X21 Y9.5 Z3.5 F1000\n').encode())
                time.sleep(10)

            time.sleep(9)

            if(y=='0'):
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='a1'):
                ser.write(('G1 X-11.5 Y32.5 Z33 F1000\n').encode())
                ser.write(('G1 X-11.5 Y27 Z36.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y23 Z39.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(12)
                ser.write(('G1 X-11.5 Y27 Z36.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y32.5 Z33 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='a2'):
                ser.write(('G1 X-12 Y28.5 Z27.5 F1000\n').encode())
                ser.write(('G1 X-12 Y23 Z31 F1000\n').encode())
                ser.write(('G1 X-12 Y18.5 Z36 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(11)
                ser.write(('G1 X-12 Y23 Z31 F1000\n').encode())
                ser.write(('G1 X-12 Y28.5 Z27.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='a3'):
                ser.write(('G1 X-13.5 Y22.5 Z24 F1000\n').encode())
                ser.write(('G1 X-13 Y16 Z29 F1000\n').encode())
                ser.write(('G1 X-13 Y13 Z32.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-13 Y16 Z29 F1000\n').encode())
                ser.write(('G1 X-13.5 Y22.5 Z24 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='a4'):
                ser.write(('G1 X-14.5 Y20.5 Z19 F1000\n').encode())
                ser.write(('G1 X-14 Y12.5 Z26 F1000\n').encode())
                ser.write(('G1 X-14 Y8.5 Z30.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-14 Y12.5 Z26 F1000\n').encode())
                ser.write(('G1 X-14.5 Y20.5 Z19 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='a5'):
                ser.write(('G1 X-15.5 Y17 Z15.5 F1000\n').encode())
                ser.write(('G1 X-15.5 Y9.5 Z22.5 F1000\n').encode())
                ser.write(('G1 X-15.5 Y4.5 Z28.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-15.5 Y9.5 Z22.5 F1000\n').encode())
                ser.write(('G1 X-15.5 Y17 Z15.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='a6'):
                ser.write(('G1 X-17 Y13.5 Z12.5 F1000\n').encode())
                ser.write(('G1 X-17 Y5.5 Z20 F1000\n').encode())
                ser.write(('G1 X-17 Y0 Z26.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-17 Y5.5 Z20 F1000\n').encode())
                ser.write(('G1 X-17 Y13.5 Z12.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='a7'):
                ser.write(('G1 X-19 Y11.5 Z9 F1000\n').encode())
                ser.write(('G1 X-19 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X-18.5 Y-3 Z24 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-19 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X-19 Y11.5 Z9 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='a8'):
                ser.write(('G1 X-22 Y6.5 Z8 F1000\n').encode())
                ser.write(('G1 X-21.5 Y0 Z14.5 F1000\n').encode())
                ser.write(('G1 X-21.5 Y-7 Z22.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-21.5 Y0 Z14.5 F1000\n').encode())
                ser.write(('G1 X-22 Y6.5 Z8 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='b1'):
                ser.write(('G1 X-9 Y29.5 Z31.5 F1000\n').encode())
                ser.write(('G1 X-8.5 Y24.5 Z36 F1000\n').encode())
                ser.write(('G1 X-8.5 Y22 Z39.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(12)
                ser.write(('G1 X-8.5 Y24.5 Z36 F1000\n').encode())
                ser.write(('G1 X-9 Y29.5 Z31.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='b2'):
                ser.write(('G1 X-9 Y26 Z26.5 F1000\n').encode())
                ser.write(('G1 X-9 Y20.5 Z30.5 F1000\n').encode())
                ser.write(('G1 X-9 Y16.5 Z35 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(11)
                ser.write(('G1 X-9 Y20.5 Z30.5 F1000\n').encode())
                ser.write(('G1 X-9 Y26 Z26.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='b3'):
                ser.write(('G1 X-9.5 Y22 Z21.5 F1000\n').encode())
                ser.write(('G1 X-9.5 Y16 Z26.5 F1000\n').encode())
                ser.write(('G1 X-9.5 Y12 Z31.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-9.5 Y16 Z26.5 F1000\n').encode())
                ser.write(('G1 X-9.5 Y22 Z21.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='b4'):
                ser.write(('G1 X-10.5 Y19 Z18 F1000\n').encode())
                ser.write(('G1 X-10.5 Y11.5 Z24.5 F1000\n').encode())
                ser.write(('G1 X-10.5 Y7.5 Z29.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-10.5 Y11.5 Z24.5 F1000\n').encode())
                ser.write(('G1 X-10.5 Y19 Z18 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='b5'):
                ser.write(('G1 X-11.5 Y17 Z14 F1000\n').encode())
                ser.write(('G1 X-11.5 Y8.5 Z20.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y3 Z27 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-11.5 Y8.5 Z20.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y17 Z14 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='b6'):
                ser.write(('G1 X-13 Y15 Z10 F1000\n').encode())
                ser.write(('G1 X-12.5 Y5 Z18 F1000\n').encode())
                ser.write(('G1 X-12.5 Y-1.5 Z25 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-12.5 Y5 Z18 F1000\n').encode())
                ser.write(('G1 X-13 Y15 Z10 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='b7'):
                ser.write(('G1 X-14.5 Y11.5 Z7 F1000\n').encode())
                ser.write(('G1 X-14 Y2 Z15.5 F1000\n').encode())
                ser.write(('G1 X-14 Y-5.5 Z23 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-14 Y2 Z15.5 F1000\n').encode())
                ser.write(('G1 X-14.5 Y11.5 Z7 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='b8'):
                ser.write(('G1 X-16.5 Y6.5 Z5.5 F1000\n').encode())
                ser.write(('G1 X-16.5 Y-1.5 Z13 F1000\n').encode())
                ser.write(('G1 X-16.5 Y-10 Z22.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-16.5 Y-1.5 Z13 F1000\n').encode())
                ser.write(('G1 X-16.5 Y6.5 Z5.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='c1'):
                ser.write(('G1 X-5.5 Y30 Z30 F1000\n').encode())
                ser.write(('G1 X-5 Y24 Z34.5 F1000\n').encode())
                ser.write(('G1 X-5.5 Y21 Z38 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(12)
                ser.write(('G1 X-5 Y24 Z34.5 F1000\n').encode())
                ser.write(('G1 X-5.5 Y30 Z30 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='c2'):
                ser.write(('G1 X-6 Y25.5 Z25 F1000\n').encode())
                ser.write(('G1 X-6 Y19.5 Z30 F1000\n').encode())
                ser.write(('G1 X-6 Y15.5 Z34 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(11)
                ser.write(('G1 X-6 Y19.5 Z30 F1000\n').encode())
                ser.write(('G1 X-6 Y25.5 Z25 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='c3'):
                ser.write(('G1 X-6 Y21 Z21 F1000\n').encode())
                ser.write(('G1 X-6 Y14.5 Z26.5 F1000\n').encode())
                ser.write(('G1 X-6 Y10.5 Z31 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-6 Y14.5 Z26.5 F1000\n').encode())
                ser.write(('G1 X-6 Y21 Z21 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='c4'):
                ser.write(('G1 X-6.5 Y18 Z17 F1000\n').encode())
                ser.write(('G1 X-6.5 Y10.5 Z23.5 F1000\n').encode())
                ser.write(('G1 X-6.5 Y6 Z29 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-6.5 Y10.5 Z23.5 F1000\n').encode())
                ser.write(('G1 X-6.5 Y18 Z17 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='c5'):
                ser.write(('G1 X-7 Y14.5 Z13.5 F1000\n').encode())
                ser.write(('G1 X-7 Y7 Z20 F1000\n').encode())
                ser.write(('G1 X-7 Y2 Z26.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-7 Y7 Z20 F1000\n').encode())
                ser.write(('G1 X-7 Y14.5 Z13.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='c6'):
                ser.write(('G1 X-8.5 Y14 Z8.5 F1000\n').encode())
                ser.write(('G1 X-8 Y3.5 Z17.5 F1000\n').encode())
                ser.write(('G1 X-8 Y-2.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-8 Y3.5 Z17.5 F1000\n').encode())
                ser.write(('G1 X-8.5 Y14 Z8.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='c7'):
                ser.write(('G1 X-9.5 Y8.5 Z6.5 F1000\n').encode())
                ser.write(('G1 X-9 Y-0.5 Z15 F1000\n').encode())
                ser.write(('G1 X-9 Y-7 Z22.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-9 Y-0.5 Z15 F1000\n').encode())
                ser.write(('G1 X-9.5 Y8.5 Z6.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='c8'):
                ser.write(('G1 X-10 Y5 Z4.5 F1000\n').encode())
                ser.write(('G1 X-10 Y-3.5 Z12 F1000\n').encode())
                ser.write(('G1 X-10.5 Y-12.5 Z22 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-10 Y-3.5 Z12 F1000\n').encode())
                ser.write(('G1 X-10 Y5 Z4.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='d1'):
                ser.write(('G1 X-3 Y28.5 Z30 F1000\n').encode())
                ser.write(('G1 X-2.5 Y23 Z34 F1000\n').encode())
                ser.write(('G1 X-2.5 Y20.5 Z37 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(12)
                ser.write(('G1 X-2.5 Y23 Z34 F1000\n').encode())
                ser.write(('G1 X-3 Y28.5 Z30 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='d2'):
                ser.write(('G1 X-2.5 Y24.5 Z24.5 F1000\n').encode())
                ser.write(('G1 X-2.5 Y19 Z29 F1000\n').encode())
                ser.write(('G1 X-2.5 Y15 Z33.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(11)
                ser.write(('G1 X-2.5 Y19 Z29 F1000\n').encode())
                ser.write(('G1 X-2.5 Y24.5 Z24.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='d3'):
                ser.write(('G1 X-2.5 Y21 Z20 F1000\n').encode())
                ser.write(('G1 X-2.5 Y14 Z26 F1000\n').encode())
                ser.write(('G1 X-2.5 Y10 Z30.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y14 Z26 F1000\n').encode())
                ser.write(('G1 X-2.5 Y21 Z20 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='d4'):
                ser.write(('G1 X-2.5 Y17.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X-2.5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X-3 Y6 Z28.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X-2.5 Y17.5 Z16.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='d5'):
                ser.write(('G1 X-3 Y14.5 Z13 F1000\n').encode())
                ser.write(('G1 X-2.5 Y7 Z19.5 F1000\n').encode())
                ser.write(('G1 X-3 Y1.5 Z26 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y7 Z19.5 F1000\n').encode())
                ser.write(('G1 X-3 Y14.5 Z13 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='d6'):
                ser.write(('G1 X-3.5 Y11.5 Z9.5 F1000\n').encode())
                ser.write(('G1 X-3 Y3 Z17 F1000\n').encode())
                ser.write(('G1 X-3 Y-3 Z24 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-3 Y3 Z17 F1000\n').encode())
                ser.write(('G1 X-3.5 Y11.5 Z9.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='d7'):
                ser.write(('G1 X-4 Y7 Z6.5 F1000\n').encode())
                ser.write(('G1 X-3.5 Y-0.5 Z13.5 F1000\n').encode())
                ser.write(('G1 X-3.5 Y-8 Z22 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-3.5 Y-0.5 Z13.5 F1000\n').encode())
                ser.write(('G1 X-4 Y7 Z6.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='d8'):
                ser.write(('G1 X-4.5 Y6 Z3 F1000\n').encode())
                ser.write(('G1 X-4 Y-7 Z15 F1000\n').encode())
                ser.write(('G1 X-4 Y-13.5 Z22 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-4 Y-7 Z15 F1000\n').encode())
                ser.write(('G1 X-4.5 Y6 Z3 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='e1'):
                ser.write(('G1 X0.5 Y28.5 Z29.5 F1000\n').encode())
                ser.write(('G1 X0.5 Y23 Z35 F1000\n').encode())
                ser.write(('G1 X0.5 Y20.5 Z37 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(12)
                ser.write(('G1 X0.5 Y23 Z35 F1000\n').encode())
                ser.write(('G1 X0.5 Y28.5 Z29.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='e2'):
                ser.write(('G1 X0.5 Y24.5 Z24.5 F1000\n').encode())
                ser.write(('G1 X0.5 Y19 Z28.5 F1000\n').encode())
                ser.write(('G1 X0.5 Y15 Z33.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(11)
                ser.write(('G1 X0.5 Y19 Z28.5 F1000\n').encode())
                ser.write(('G1 X0.5 Y24.5 Z24.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='e3'):
                ser.write(('G1 X1 Y21 Z20 F1000\n').encode())
                ser.write(('G1 X1 Y14 Z25.5 F1000\n').encode())
                ser.write(('G1 X1 Y10 Z30 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X1 Y14 Z25.5 F1000\n').encode())
                ser.write(('G1 X1 Y21 Z20 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='e4'):
                ser.write(('G1 X1 Y17.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X1.5 Y10 Z23 F1000\n').encode())
                ser.write(('G1 X1.5 Y5.5 Z28 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X1.5 Y10 Z23 F1000\n').encode())
                ser.write(('G1 X1 Y17.5 Z16.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='e5'):
                ser.write(('G1 X1 Y15 Z12 F1000\n').encode())
                ser.write(('G1 X1.5 Y7 Z19 F1000\n').encode())
                ser.write(('G1 X1.5 Y1 Z26 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X1.5 Y7 Z19 F1000\n').encode())
                ser.write(('G1 X1 Y15 Z12 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='e6'):
                ser.write(('G1 X1.5 Y11 Z9.5 F1000\n').encode())
                ser.write(('G1 X2 Y3 Z16.5 F1000\n').encode())
                ser.write(('G1 X2 Y-3 Z24 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X2 Y3 Z16.5 F1000\n').encode())
                ser.write(('G1 X1.5 Y11 Z9.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='e7'):
                ser.write(('G1 X2 Y7 Z6.5 F1000\n').encode())
                ser.write(('G1 X2 Y0 Z13 F1000\n').encode())
                ser.write(('G1 X2 Y-8 Z22 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X2 Y0 Z13 F1000\n').encode())
                ser.write(('G1 X2 Y7 Z6.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='e8'):
                ser.write(('G1 X2.5 Y4.5 Z3.5 F1000\n').encode())
                ser.write(('G1 X3 Y-6 Z13 F1000\n').encode())
                ser.write(('G1 X3 Y-14 Z21.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X3 Y-6 Z13 F1000\n').encode())
                ser.write(('G1 X2.5 Y4.5 Z3.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='f1'):
                ser.write(('G1 X3.5 Y29 Z30 F1000\n').encode())
                ser.write(('G1 X3.5 Y23.5 Z34 F1000\n').encode())
                ser.write(('G1 X3.5 Y20.5 Z37.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(12)
                ser.write(('G1 X3.5 Y23.5 Z34 F1000\n').encode())
                ser.write(('G1 X3.5 Y29 Z30 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='f2'):
                ser.write(('G1 X4 Y25 Z25 F1000\n').encode())
                ser.write(('G1 X4 Y19 Z29.5 F1000\n').encode())
                ser.write(('G1 X4 Y15.5 Z33.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(11)
                ser.write(('G1 X4 Y19 Z29.5 F1000\n').encode())
                ser.write(('G1 X4 Y25 Z25 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='f3'):
                ser.write(('G1 X4.5 Y21.5 Z20 F1000\n').encode())
                ser.write(('G1 X4.5 Y14.5 Z25.5 F1000\n').encode())
                ser.write(('G1 X4.5 Y11 Z30 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X4.5 Y14.5 Z25.5 F1000\n').encode())
                ser.write(('G1 X4.5 Y21.5 Z20 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='f4'):
                ser.write(('G1 X5 Y19 Z16 F1000\n').encode())
                ser.write(('G1 X5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X5 Y6 Z28 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X5 Y19 Z16 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='f5'):
                ser.write(('G1 X5.5 Y15.5 Z12.5 F1000\n').encode())
                ser.write(('G1 X6 Y7.5 Z19 F1000\n').encode())
                ser.write(('G1 X6 Y1.5 Z26 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X6 Y7.5 Z19 F1000\n').encode())
                ser.write(('G1 X5.5 Y15.5 Z12.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='f6'):
                ser.write(('G1 X6.5 Y11.5 Z9.5 F1000\n').encode())
                ser.write(('G1 X7 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X7 Y-2.5 Z24 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X7 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X6.5 Y11.5 Z9.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='f7'):
                ser.write(('G1 X7.5 Y6 Z8 F1000\n').encode())
                ser.write(('G1 X8 Y-0.5 Z14 F1000\n').encode())
                ser.write(('G1 X8 Y-7.5 Z22 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X8 Y-0.5 Z14 F1000\n').encode())
                ser.write(('G1 X7.5 Y6 Z8 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='f8'):
                ser.write(('G1 X8.5 Y7.5 Z2 F1000\n').encode())
                ser.write(('G1 X9 Y-4 Z12 F1000\n').encode())
                ser.write(('G1 X9 Y-12.5 Z21 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X9 Y-4 Z12 F1000\n').encode())
                ser.write(('G1 X8.5 Y7.5 Z2 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='g1'):
                ser.write(('G1 X6 Y30 Z31.5 F1000\n').encode())
                ser.write(('G1 X6 Y25 Z35 F1000\n').encode())
                ser.write(('G1 X6 Y21.5 Z37.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(12)
                ser.write(('G1 X6 Y25 Z35 F1000\n').encode())
                ser.write(('G1 X6 Y30 Z31.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='g2'):
                ser.write(('G1 X7 Y25 Z25.5 F1000\n').encode())
                ser.write(('G1 X7 Y20 Z29.5 F1000\n').encode())
                ser.write(('G1 X7 Y16.5 Z34 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(11)
                ser.write(('G1 X7 Y20 Z29.5 F1000\n').encode())
                ser.write(('G1 X7 Y25 Z25.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='g3'):
                ser.write(('G1 X8 Y21.5 Z21 F1000\n').encode())
                ser.write(('G1 X8 Y15 Z26.5 F1000\n').encode())
                ser.write(('G1 X8 Y11.5 Z30.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X8 Y15 Z26.5 F1000\n').encode())
                ser.write(('G1 X8 Y21.5 Z21 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='g4'):
                ser.write(('G1 X9 Y18.5 Z17 F1000\n').encode())
                ser.write(('G1 X9 Y11.5 Z23 F1000\n').encode())
                ser.write(('G1 X9 Y6.5 Z28.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X9 Y11.5 Z23 F1000\n').encode())
                ser.write(('G1 X9 Y18.5 Z17 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='g5'):
                ser.write(('G1 X9.5 Y15.5 Z13.5 F1000\n').encode())
                ser.write(('G1 X10 Y8 Z20 F1000\n').encode())
                ser.write(('G1 X10 Y2.5 Z26 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X10 Y8 Z20 F1000\n').encode())
                ser.write(('G1 X9.5 Y15.5 Z13.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='g6'):
                ser.write(('G1 X11.5 Y12 Z10.5 F1000\n').encode())
                ser.write(('G1 X11.5 Y4 Z17.5 F1000\n').encode())
                ser.write(('G1 X11.5 Y-1.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X11.5 Y4 Z17.5 F1000\n').encode())
                ser.write(('G1 X11.5 Y12 Z10.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='g7'):
                ser.write(('G1 X13 Y7.5 Z8 F1000\n').encode())
                ser.write(('G1 X13 Y0.5 Z14.5 F1000\n').encode())
                ser.write(('G1 X13 Y-6.5 Z22.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X13 Y0.5 Z14.5 F1000\n').encode())
                ser.write(('G1 X13 Y7.5 Z8 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='g8'):
                ser.write(('G1 X16 Y6 Z4.5 F1000\n').encode())
                ser.write(('G1 X16 Y-2 Z12 F1000\n').encode())
                ser.write(('G1 X16 Y-8.5 Z19 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X16 Y-2 Z12 F1000\n').encode())
                ser.write(('G1 X16 Y6 Z4.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='h1'):
                ser.write(('G1 X9 Y30.5 Z32 F1000\n').encode())
                ser.write(('G1 X9 Y25.5 Z35.5 F1000\n').encode())
                ser.write(('G1 X9 Y23 Z39 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(12)
                ser.write(('G1 X9 Y25.5 Z35.5 F1000\n').encode())
                ser.write(('G1 X9 Y30.5 Z32 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='h2'):
                ser.write(('G1 X10 Y26 Z26.5 F1000\n').encode())
                ser.write(('G1 X10 Y21 Z30.5 F1000\n').encode())
                ser.write(('G1 X10 Y17.5 Z35 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(11)
                ser.write(('G1 X10 Y21 Z30.5 F1000\n').encode())
                ser.write(('G1 X10 Y26 Z26.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='h3'):
                ser.write(('G1 X11 Y21.5 Z22.5 F1000\n').encode())
                ser.write(('G1 X11 Y17 Z26 F1000\n').encode())
                ser.write(('G1 X11.5 Y13 Z31 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X11 Y17 Z26 F1000\n').encode())
                ser.write(('G1 X11 Y21.5 Z22.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='h4'):
                ser.write(('G1 X12.5 Y19.5 Z17.5 F1000\n').encode())
                ser.write(('G1 X12.5 Y13 Z23 F1000\n').encode())
                ser.write(('G1 X12.5 Y8.5 Z29 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X12.5 Y13 Z23 F1000\n').encode())
                ser.write(('G1 X12.5 Y19.5 Z17.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='h5'):
                ser.write(('G1 X13.5 Y17.5 Z14 F1000\n').encode())
                ser.write(('G1 X14 Y9 Z21 F1000\n').encode())
                ser.write(('G1 X14 Y4 Z27 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X14 Y9 Z21 F1000\n').encode())
                ser.write(('G1 X13.5 Y17.5 Z14 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='h6'):
                ser.write(('G1 X15.5 Y14 Z11 F1000\n').encode())
                ser.write(('G1 X16 Y6 Z18 F1000\n').encode())
                ser.write(('G1 X16 Y0.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X16 Y6 Z18 F1000\n').encode())
                ser.write(('G1 X15.5 Y14 Z11 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='h7'):
                ser.write(('G1 X17.5 Y9.5 Z8.5 F1000\n').encode())
                ser.write(('G1 X18 Y2 Z16 F1000\n').encode())
                ser.write(('G1 X18 Y-4 Z22.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X18 Y2 Z16 F1000\n').encode())
                ser.write(('G1 X17.5 Y9.5 Z8.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='h8'):
                ser.write(('G1 X21 Y9.5 Z3.5 F1000\n').encode())
                ser.write(('G1 X21 Y1.5 Z11.5 F1000\n').encode())
                ser.write(('G1 X21.5 Y-6.5 Z20 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X21 Y1.5 Z11.5 F1000\n').encode())
                ser.write(('G1 X21 Y9.5 Z3.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())

            time.sleep(5)

        else:
            roszada = x+y
            if(roszada=='O-O'):
                print("Została wykonana krotka roszada O-O")
                time.sleep(2)
                ser.write(('G1 X2.5 Y4.5 Z3.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X3 Y-6 Z13 F1000\n').encode())
                ser.write(('G1 X3 Y-14 Z21.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(5)
                ser.write(('G1 X3 Y-6 Z13 F1000\n').encode())
                ser.write(('G1 X2.5 Y4.5 Z3.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X16 Y6 Z4.5 F1000\n').encode())
                ser.write(('G1 X16 Y-2 Z12 F1000\n').encode())
                ser.write(('G1 X16 Y-8.5 Z19 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(5)
                ser.write(('G1 X16 Y-2 Z12 F1000\n').encode())
                ser.write(('G1 X16 Y6 Z4.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X21 Y9.5 Z3.5 F1000\n').encode())
                ser.write(('G1 X21 Y1.5 Z11.5 F1000\n').encode())
                ser.write(('G1 X21.5 Y-6.5 Z20 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(5)
                ser.write(('G1 X21 Y1.5 Z11.5 F1000\n').encode())
                ser.write(('G1 X21 Y9.5 Z3.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X8.5 Y7.5 Z2 F1000\n').encode())
                ser.write(('G1 X9 Y-4 Z12 F1000\n').encode())
                ser.write(('G1 X9 Y-12.5 Z21 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X9 Y-4 Z12 F1000\n').encode())
                ser.write(('G1 X8.5 Y7.5 Z2 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(roszada=='O-O-O'):
                time.sleep(2)
                print("została wykonana długa roszada O-O-O")
                ser.write(('G1 X2.5 Y4.5 Z3.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X3 Y-6 Z13 F1000\n').encode())
                ser.write(('G1 X3 Y-14 Z21.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(5)
                ser.write(('G1 X3 Y-6 Z13 F1000\n').encode())
                ser.write(('G1 X2.5 Y4.5 Z3.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X-10 Y5 Z4.5 F1000\n').encode())
                ser.write(('G1 X-10 Y-3.5 Z12 F1000\n').encode())
                ser.write(('G1 X-10.5 Y-12.5 Z22 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(5)
                ser.write(('G1 X-10 Y-3.5 Z12 F1000\n').encode())
                ser.write(('G1 X-10 Y5 Z4.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X-22 Y6.5 Z8 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-21.5 Y0 Z14.5 F1000\n').encode())
                ser.write(('G1 X-21.5 Y-7 Z22.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(5)
                ser.write(('G1 X-21.5 Y0 Z14.5 F1000\n').encode())
                ser.write(('G1 X-22 Y6.5 Z8 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X-4.5 Y6 Z3 F1000\n').encode())
                ser.write(('G1 X-4 Y-7 Z15 F1000\n').encode())
                ser.write(('G1 X-4 Y-13.5 Z22 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(5)
                ser.write(('G1 X-4 Y-7 Z15 F1000\n').encode())
                ser.write(('G1 X-4.5 Y6 Z3 F1000\n').encode())
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
                time.sleep(5)
            elif(x=='a1'):
                ser.write(('G1 X-11.5 Y32.5 Z33 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-11.5 Y27 Z36.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y23 Z39.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-11.5 Y27 Z36.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y32.5 Z33 F1000\n').encode())
                time.sleep(5)
            elif(x=='a2'):
                ser.write(('G1 X-12 Y28.5 Z27.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-12 Y23 Z31 F1000\n').encode())
                ser.write(('G1 X-12 Y18.5 Z36 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-12 Y23 Z31 F1000\n').encode())
                ser.write(('G1 X-12 Y28.5 Z27.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='a3'):
                ser.write(('G1 X-13.5 Y22.5 Z24 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-13 Y16 Z29 F1000\n').encode())
                ser.write(('G1 X-13 Y13 Z32.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-13 Y16 Z29 F1000\n').encode())
                ser.write(('G1 X-13.5 Y22.5 Z24 F1000\n').encode())
                time.sleep(5)
            elif(x=='a4'):
                ser.write(('G1 X-14.5 Y20.5 Z19 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-14 Y12.5 Z26 F1000\n').encode())
                ser.write(('G1 X-14 Y8.5 Z30.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-14 Y12.5 Z26 F1000\n').encode())
                ser.write(('G1 X-14.5 Y20.5 Z19 F1000\n').encode())
                time.sleep(5)
            elif(x=='a5'):
                ser.write(('G1 X-15.5 Y17 Z15.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-15.5 Y9.5 Z22.5 F1000\n').encode())
                ser.write(('G1 X-15.5 Y4.5 Z28.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(3)
                ser.write(('G1 X-15.5 Y9.5 Z22.5 F1000\n').encode())
                ser.write(('G1 X-15.5 Y17 Z15.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='a6'):
                ser.write(('G1 X-17 Y13.5 Z12.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-17 Y5.5 Z20 F1000\n').encode())
                ser.write(('G1 X-17 Y0 Z26.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-17 Y5.5 Z20 F1000\n').encode())
                ser.write(('G1 X-17 Y13.5 Z12.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='a7'):
                ser.write(('G1 X-19 Y11.5 Z9 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-19 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X-18.5 Y-3 Z24 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-19 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X-19 Y11.5 Z9 F1000\n').encode())
                time.sleep(5)
            elif(x=='a8'):
                ser.write(('G1 X-22 Y6.5 Z8 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-21.5 Y0 Z14.5 F1000\n').encode())
                ser.write(('G1 X-21.5 Y-7 Z22.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-21.5 Y0 Z14.5 F1000\n').encode())
                ser.write(('G1 X-22 Y6.5 Z8 F1000\n').encode())
                time.sleep(5)
            elif(x=='b1'):
                ser.write(('G1 X-9 Y29.5 Z31.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-8.5 Y24.5 Z36 F1000\n').encode())
                ser.write(('G1 X-8.5 Y22 Z39.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-8.5 Y24.5 Z36 F1000\n').encode())
                ser.write(('G1 X-9 Y29.5 Z31.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='b2'):
                ser.write(('G1 X-9 Y26 Z26.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-9 Y20.5 Z30.5 F1000\n').encode())
                ser.write(('G1 X-9 Y16.5 Z35 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-9 Y20.5 Z30.5 F1000\n').encode())
                ser.write(('G1 X-9 Y26 Z26.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='b3'):
                ser.write(('G1 X-9.5 Y22 Z21.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-9.5 Y16 Z26.5 F1000\n').encode())
                ser.write(('G1 X-9.5 Y12 Z31.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-9.5 Y16 Z26.5 F1000\n').encode())
                ser.write(('G1 X-9.5 Y22 Z21.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='b4'):
                ser.write(('G1 X-10.5 Y19 Z18 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-10.5 Y11.5 Z24.5 F1000\n').encode())
                ser.write(('G1 X-10.5 Y7.5 Z29.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-10.5 Y11.5 Z24.5 F1000\n').encode())
                ser.write(('G1 X-10.5 Y19 Z18 F1000\n').encode())
                time.sleep(5)
            elif(x=='b5'):
                ser.write(('G1 X-11.5 Y17 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-11.5 Y8.5 Z20.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y3 Z27 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-11.5 Y8.5 Z20.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y17 Z14 F1000\n').encode())
                time.sleep(5)
            elif(x=='b6'):
                ser.write(('G1 X-13 Y15 Z10 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-12.5 Y5 Z18 F1000\n').encode())
                ser.write(('G1 X-12.5 Y-1.5 Z25 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-12.5 Y5 Z18 F1000\n').encode())
                ser.write(('G1 X-13 Y15 Z10 F1000\n').encode())
                time.sleep(5)
            elif(x=='b7'):
                ser.write(('G1 X-14.5 Y11.5 Z7 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-14 Y2 Z15.5 F1000\n').encode())
                ser.write(('G1 X-14 Y-5.5 Z23 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-14 Y2 Z15.5 F1000\n').encode())
                ser.write(('G1 X-14.5 Y11.5 Z7 F1000\n').encode())
                time.sleep(5)
            elif(x=='b8'):
                ser.write(('G1 X-16.5 Y6.5 Z5.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-16.5 Y-1.5 Z13 F1000\n').encode())
                ser.write(('G1 X-16.5 Y-10 Z22.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-16.5 Y-1.5 Z13 F1000\n').encode())
                ser.write(('G1 X-16.5 Y6.5 Z5.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='c1'):
                ser.write(('G1 X-5.5 Y30 Z30 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-5 Y24 Z34.5 F1000\n').encode())
                ser.write(('G1 X-5.5 Y21 Z38 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-5 Y24 Z34.5 F1000\n').encode())
                ser.write(('G1 X-5.5 Y30 Z30 F1000\n').encode())
                time.sleep(5)
            elif(x=='c2'):
                ser.write(('G1 X-6 Y25.5 Z25 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-6 Y19.5 Z30 F1000\n').encode())
                ser.write(('G1 X-6 Y15.5 Z34 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-6 Y19.5 Z30 F1000\n').encode())
                ser.write(('G1 X-6 Y25.5 Z25 F1000\n').encode())
                time.sleep(5)
            elif(x=='c3'):
                ser.write(('G1 X-6 Y21 Z21 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-6 Y14.5 Z26.5 F1000\n').encode())
                ser.write(('G1 X-6 Y10.5 Z31 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-6 Y14.5 Z26.5 F1000\n').encode())
                ser.write(('G1 X-6 Y21 Z21 F1000\n').encode())
                time.sleep(5)
            elif(x=='c4'):
                ser.write(('G1 X-6.5 Y18 Z17 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-6.5 Y10.5 Z23.5 F1000\n').encode())
                ser.write(('G1 X-6.5 Y6 Z29 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-6.5 Y10.5 Z23.5 F1000\n').encode())
                ser.write(('G1 X-6.5 Y18 Z17 F1000\n').encode())
                time.sleep(5)
            elif(x=='c5'):
                ser.write(('G1 X-7 Y14.5 Z13.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-7 Y7 Z20 F1000\n').encode())
                ser.write(('G1 X-7 Y2 Z26.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-7 Y7 Z20 F1000\n').encode())
                ser.write(('G1 X-7 Y14.5 Z13.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='c6'):
                ser.write(('G1 X-8.5 Y14 Z8.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-8 Y3.5 Z17.5 F1000\n').encode())
                ser.write(('G1 X-8 Y-2.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-8 Y3.5 Z17.5 F1000\n').encode())
                ser.write(('G1 X-8.5 Y14 Z8.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='c7'):
                ser.write(('G1 X-9.5 Y8.5 Z6.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-9 Y-0.5 Z15 F1000\n').encode())
                ser.write(('G1 X-9 Y-7 Z22.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-9 Y-0.5 Z15 F1000\n').encode())
                ser.write(('G1 X-9.5 Y8.5 Z6.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='c8'):
                ser.write(('G1 X-10 Y5 Z4.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-10 Y-3.5 Z12 F1000\n').encode())
                ser.write(('G1 X-10.5 Y-12.5 Z22 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-10 Y-3.5 Z12 F1000\n').encode())
                ser.write(('G1 X-10 Y5 Z4.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='d1'):
                ser.write(('G1 X-3 Y28.5 Z30 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-2.5 Y23 Z34 F1000\n').encode())
                ser.write(('G1 X-2.5 Y20.5 Z37 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y23 Z34 F1000\n').encode())
                ser.write(('G1 X-3 Y28.5 Z30 F1000\n').encode())
                time.sleep(5)
            elif(x=='d2'):
                ser.write(('G1 X-2.5 Y24.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-2.5 Y19 Z29 F1000\n').encode())
                ser.write(('G1 X-2.5 Y15 Z33.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y19 Z29 F1000\n').encode())
                ser.write(('G1 X-2.5 Y24.5 Z24.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='d3'):
                ser.write(('G1 X-2.5 Y21 Z20 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-2.5 Y14 Z26 F1000\n').encode())
                ser.write(('G1 X-2.5 Y10 Z30.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y14 Z26 F1000\n').encode())
                ser.write(('G1 X-2.5 Y21 Z20 F1000\n').encode())
                time.sleep(5)
            elif(x=='d4'):
                ser.write(('G1 X-2.5 Y17.5 Z16.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-2.5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X-3 Y6 Z28.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X-2.5 Y17.5 Z16.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='d5'):
                ser.write(('G1 X-3 Y14.5 Z13 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-2.5 Y7 Z19.5 F1000\n').encode())
                ser.write(('G1 X-3 Y1.5 Z26 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-2.5 Y7 Z19.5 F1000\n').encode())
                ser.write(('G1 X-3 Y14.5 Z13 F1000\n').encode())
                time.sleep(5)
            elif(x=='d6'):
                ser.write(('G1 X-3.5 Y11.5 Z9.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-3 Y3 Z17 F1000\n').encode())
                ser.write(('G1 X-3 Y-3 Z24 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-3 Y3 Z17 F1000\n').encode())
                ser.write(('G1 X-3.5 Y11.5 Z9.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='d7'):
                ser.write(('G1 X-4 Y7 Z6.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-3.5 Y-0.5 Z13.5 F1000\n').encode())
                ser.write(('G1 X-3.5 Y-8 Z22 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-3.5 Y-0.5 Z13.5 F1000\n').encode())
                ser.write(('G1 X-4 Y7 Z6.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='d8'):
                ser.write(('G1 X-4.5 Y6 Z3 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-4 Y-7 Z15 F1000\n').encode())
                ser.write(('G1 X-4 Y-13.5 Z22 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X-4 Y-7 Z15 F1000\n').encode())
                ser.write(('G1 X-4.5 Y6 Z3 F1000\n').encode())
                time.sleep(5)
            elif(x=='e1'):
                ser.write(('G1 X0.5 Y28.5 Z29.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X0.5 Y23 Z35 F1000\n').encode())
                ser.write(('G1 X0.5 Y20.5 Z37 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X0.5 Y23 Z35 F1000\n').encode())
                ser.write(('G1 X0.5 Y28.5 Z29.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='e2'):
                ser.write(('G1 X0.5 Y24.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X0.5 Y19 Z28.5 F1000\n').encode())
                ser.write(('G1 X0.5 Y15 Z33.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X0.5 Y19 Z28.5 F1000\n').encode())
                ser.write(('G1 X0.5 Y24.5 Z24.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='e3'):
                ser.write(('G1 X1 Y21 Z20 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X1 Y14 Z25.5 F1000\n').encode())
                ser.write(('G1 X1 Y10 Z30 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X1 Y14 Z25.5 F1000\n').encode())
                ser.write(('G1 X1 Y21 Z20 F1000\n').encode())
                time.sleep(5)
            elif(x=='e4'):
                ser.write(('G1 X1 Y17.5 Z16.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X1.5 Y10 Z23 F1000\n').encode())
                ser.write(('G1 X1.5 Y5.5 Z28 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X1.5 Y10 Z23 F1000\n').encode())
                ser.write(('G1 X1 Y17.5 Z16.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='e5'):
                ser.write(('G1 X1 Y15 Z12 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X1.5 Y7 Z19 F1000\n').encode())
                ser.write(('G1 X1.5 Y1 Z26 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X1.5 Y7 Z19 F1000\n').encode())
                ser.write(('G1 X1 Y15 Z12 F1000\n').encode())
                time.sleep(5)
            elif(x=='e6'):
                ser.write(('G1 X1.5 Y11 Z9.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X2 Y3 Z16.5 F1000\n').encode())
                ser.write(('G1 X2 Y-3 Z24 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X2 Y3 Z16.5 F1000\n').encode())
                ser.write(('G1 X1.5 Y11 Z9.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='e7'):
                ser.write(('G1 X2 Y7 Z6.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X2 Y0 Z13 F1000\n').encode())
                ser.write(('G1 X2 Y-8 Z22 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X2 Y0 Z13 F1000\n').encode())
                ser.write(('G1 X2 Y7 Z6.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='e8'):
                ser.write(('G1 X2.5 Y4.5 Z3.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X3 Y-6 Z13 F1000\n').encode())
                ser.write(('G1 X3 Y-14 Z21.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X3 Y-6 Z13 F1000\n').encode())
                ser.write(('G1 X2.5 Y4.5 Z3.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='f1'):
                ser.write(('G1 X3.5 Y29 Z30 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X3.5 Y23.5 Z34 F1000\n').encode())
                ser.write(('G1 X3.5 Y20.5 Z37.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X3.5 Y23.5 Z34 F1000\n').encode())
                ser.write(('G1 X3.5 Y29 Z30 F1000\n').encode())
                time.sleep(5)
            elif(x=='f2'):
                ser.write(('G1 X4 Y25 Z25 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X4 Y19 Z29.5 F1000\n').encode())
                ser.write(('G1 X4 Y15.5 Z33.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X4 Y19 Z29.5 F1000\n').encode())
                ser.write(('G1 X4 Y25 Z25 F1000\n').encode())
                time.sleep(5)
            elif(x=='f3'):
                ser.write(('G1 X4.5 Y21.5 Z20 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X4.5 Y14.5 Z25.5 F1000\n').encode())
                ser.write(('G1 X4.5 Y11 Z30 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X4.5 Y14.5 Z25.5 F1000\n').encode())
                ser.write(('G1 X4.5 Y21.5 Z20 F1000\n').encode())
                time.sleep(5)
            elif(x=='f4'):
                ser.write(('G1 X5 Y19 Z16 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X5 Y6 Z28 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X5 Y19 Z16 F1000\n').encode())
                time.sleep(5)
            elif(x=='f5'):
                ser.write(('G1 X5.5 Y15.5 Z12.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X6 Y7.5 Z19 F1000\n').encode())
                ser.write(('G1 X6 Y1.5 Z26 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X6 Y7.5 Z19 F1000\n').encode())
                ser.write(('G1 X5.5 Y15.5 Z12.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='f6'):
                ser.write(('G1 X6.5 Y11.5 Z9.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X7 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X7 Y-2.5 Z24 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X7 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X6.5 Y11.5 Z9.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='f7'):
                ser.write(('G1 X7.5 Y6 Z8 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X8 Y-0.5 Z14 F1000\n').encode())
                ser.write(('G1 X8 Y-7.5 Z22 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X8 Y-0.5 Z14 F1000\n').encode())
                ser.write(('G1 X7.5 Y6 Z8 F1000\n').encode())
                time.sleep(5)
            elif(x=='f8'):
                ser.write(('G1 X8.5 Y7.5 Z2 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X9 Y-4 Z12 F1000\n').encode())
                ser.write(('G1 X9 Y-12.5 Z21 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X9 Y-4 Z12 F1000\n').encode())
                ser.write(('G1 X8.5 Y7.5 Z2 F1000\n').encode())
                time.sleep(5)
            elif(x=='g1'):
                ser.write(('G1 X6 Y30 Z31.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X6 Y25 Z35 F1000\n').encode())
                ser.write(('G1 X6 Y21.5 Z37.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X6 Y25 Z35 F1000\n').encode())
                ser.write(('G1 X6 Y30 Z31.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='g2'):
                ser.write(('G1 X7 Y25 Z25.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X7 Y20 Z29.5 F1000\n').encode())
                ser.write(('G1 X7 Y16.5 Z34 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X7 Y20 Z29.5 F1000\n').encode())
                ser.write(('G1 X7 Y25 Z25.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='g3'):
                ser.write(('G1 X8 Y21.5 Z21 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X8 Y15 Z26.5 F1000\n').encode())
                ser.write(('G1 X8 Y11.5 Z30.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X8 Y15 Z26.5 F1000\n').encode())
                ser.write(('G1 X8 Y21.5 Z21 F1000\n').encode())
                time.sleep(5)
            elif(x=='g4'):
                ser.write(('G1 X9 Y18.5 Z17 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X9 Y11.5 Z23 F1000\n').encode())
                ser.write(('G1 X9 Y6.5 Z28.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X9 Y11.5 Z23 F1000\n').encode())
                ser.write(('G1 X9 Y18.5 Z17 F1000\n').encode())
                time.sleep(5)
            elif(x=='g5'):
                ser.write(('G1 X9.5 Y15.5 Z13.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X10 Y8 Z20 F1000\n').encode())
                ser.write(('G1 X10 Y2.5 Z26 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X10 Y8 Z20 F1000\n').encode())
                ser.write(('G1 X9.5 Y15.5 Z13.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='g6'):
                ser.write(('G1 X11.5 Y12 Z10.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X11.5 Y4 Z17.5 F1000\n').encode())
                ser.write(('G1 X11.5 Y-1.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X11.5 Y4 Z17.5 F1000\n').encode())
                ser.write(('G1 X11.5 Y12 Z10.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='g7'):
                ser.write(('G1 X13 Y7.5 Z8 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X13 Y0.5 Z14.5 F1000\n').encode())
                ser.write(('G1 X13 Y-6.5 Z22.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X13 Y0.5 Z14.5 F1000\n').encode())
                ser.write(('G1 X13 Y7.5 Z8 F1000\n').encode())
                time.sleep(5)
            elif(x=='g8'):
                ser.write(('G1 X16 Y6 Z4.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X16 Y-2 Z12 F1000\n').encode())
                ser.write(('G1 X16 Y-8.5 Z19 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X16 Y-2 Z12 F1000\n').encode())
                ser.write(('G1 X16 Y6 Z4.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='h1'):
                ser.write(('G1 X9 Y30.5 Z32 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X9 Y25.5 Z35.5 F1000\n').encode())
                ser.write(('G1 X9 Y23 Z39 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X9 Y25.5 Z35.5 F1000\n').encode())
                ser.write(('G1 X9 Y30.5 Z32 F1000\n').encode())
                time.sleep(5)
            elif(x=='h2'):
                ser.write(('G1 X10 Y26 Z26.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X10 Y21 Z30.5 F1000\n').encode())
                ser.write(('G1 X10 Y17.5 Z35 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X10 Y21 Z30.5 F1000\n').encode())
                ser.write(('G1 X10 Y26 Z26.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='h3'):
                ser.write(('G1 X11 Y21.5 Z22.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X11 Y17 Z26 F1000\n').encode())
                ser.write(('G1 X11.5 Y13 Z31 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X11 Y17 Z26 F1000\n').encode())
                ser.write(('G1 X11 Y21.5 Z22.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='h4'):
                ser.write(('G1 X12.5 Y19.5 Z17.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X12.5 Y13 Z23 F1000\n').encode())
                ser.write(('G1 X12.5 Y8.5 Z29 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X12.5 Y13 Z23 F1000\n').encode())
                ser.write(('G1 X12.5 Y19.5 Z17.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='h5'):
                ser.write(('G1 X13.5 Y17.5 Z14 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X14 Y9 Z21 F1000\n').encode())
                ser.write(('G1 X14 Y4 Z27 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X14 Y9 Z21 F1000\n').encode())
                ser.write(('G1 X13.5 Y17.5 Z14 F1000\n').encode())
                time.sleep(5)
            elif(x=='h6'):
                ser.write(('G1 X15.5 Y14 Z11 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X16 Y6 Z18 F1000\n').encode())
                ser.write(('G1 X16 Y0.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X16 Y6 Z18 F1000\n').encode())
                ser.write(('G1 X15.5 Y14 Z11 F1000\n').encode())
                time.sleep(5)
            elif(x=='h7'):
                ser.write(('G1 X17.5 Y9.5 Z8.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X18 Y2 Z16 F1000\n').encode())
                ser.write(('G1 X18 Y-4 Z22.5 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                ser.write(('G1 X18 Y2 Z16 F1000\n').encode())
                ser.write(('G1 X17.5 Y9.5 Z8.5 F1000\n').encode())
                time.sleep(5)
            elif(x=='h8'):
                ser.write(('G1 X21 Y9.5 Z3.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X21 Y1.5 Z11.5 F1000\n').encode())
                ser.write(('G1 X21.5 Y-6.5 Z20 F1000\n').encode())
                ser.write(('M3 S80\n').encode()) #zamkniecie
                time.sleep(7)
                ser.write(('G1 X21 Y1.5 Z11.5 F1000\n').encode())
                ser.write(('G1 X21 Y9.5 Z3.5 F1000\n').encode())
                time.sleep(5)

            time.sleep(2)

            if(y=='0'):
                xyz=1
            elif(y=='a1'):
                ser.write(('G1 X-11.5 Y32.5 Z33 F1000\n').encode())
                ser.write(('G1 X-11.5 Y27 Z36.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y23 Z39.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-11.5 Y27 Z36.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y32.5 Z33 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())

            elif(y=='a2'):
                ser.write(('G1 X-12 Y28.5 Z27.5 F1000\n').encode())
                ser.write(('G1 X-12 Y23 Z31 F1000\n').encode())
                ser.write(('G1 X-12 Y18.5 Z36 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-12 Y23 Z31 F1000\n').encode())
                ser.write(('G1 X-12 Y28.5 Z27.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())

            elif(y=='a3'):
                ser.write(('G1 X-13.5 Y22.5 Z24 F1000\n').encode())
                ser.write(('G1 X-13 Y16 Z29 F1000\n').encode())
                ser.write(('G1 X-13 Y13 Z32.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-13 Y16 Z29 F1000\n').encode())
                ser.write(('G1 X-13.5 Y22.5 Z24 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='a4'):
                ser.write(('G1 X-14.5 Y20.5 Z19 F1000\n').encode())
                ser.write(('G1 X-14 Y12.5 Z26 F1000\n').encode())
                ser.write(('G1 X-14 Y8.5 Z30.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-14 Y12.5 Z26 F1000\n').encode())
                ser.write(('G1 X-14.5 Y20.5 Z19 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='a5'):
                ser.write(('G1 X-15.5 Y17 Z15.5 F1000\n').encode())
                ser.write(('G1 X-15.5 Y9.5 Z22.5 F1000\n').encode())
                ser.write(('G1 X-15.5 Y4.5 Z28.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(3)
                ser.write(('G1 X-15.5 Y9.5 Z22.5 F1000\n').encode())
                ser.write(('G1 X-15.5 Y17 Z15.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='a6'):
                ser.write(('G1 X-17 Y13.5 Z12.5 F1000\n').encode())
                ser.write(('G1 X-17 Y5.5 Z20 F1000\n').encode())
                ser.write(('G1 X-17 Y0 Z26.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-17 Y5.5 Z20 F1000\n').encode())
                ser.write(('G1 X-17 Y13.5 Z12.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='a7'):
                ser.write(('G1 X-19 Y11.5 Z9 F1000\n').encode())
                ser.write(('G1 X-19 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X-18.5 Y-3 Z24 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-19 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X-19 Y11.5 Z9 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='a8'):
                ser.write(('G1 X-22 Y6.5 Z8 F1000\n').encode())
                ser.write(('G1 X-21.5 Y0 Z14.5 F1000\n').encode())
                ser.write(('G1 X-21.5 Y-7 Z22.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-21.5 Y0 Z14.5 F1000\n').encode())
                ser.write(('G1 X-22 Y6.5 Z8 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='b1'):
                ser.write(('G1 X-9 Y29.5 Z31.5 F1000\n').encode())
                ser.write(('G1 X-8.5 Y24.5 Z36 F1000\n').encode())
                ser.write(('G1 X-8.5 Y22 Z39.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-8.5 Y24.5 Z36 F1000\n').encode())
                ser.write(('G1 X-9 Y29.5 Z31.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='b2'):
                ser.write(('G1 X-9 Y26 Z26.5 F1000\n').encode())
                ser.write(('G1 X-9 Y20.5 Z30.5 F1000\n').encode())
                ser.write(('G1 X-9 Y16.5 Z35 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-9 Y20.5 Z30.5 F1000\n').encode())
                ser.write(('G1 X-9 Y26 Z26.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='b3'):
                ser.write(('G1 X-9.5 Y22 Z21.5 F1000\n').encode())
                ser.write(('G1 X-9.5 Y16 Z26.5 F1000\n').encode())
                ser.write(('G1 X-9.5 Y12 Z31.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-9.5 Y16 Z26.5 F1000\n').encode())
                ser.write(('G1 X-9.5 Y22 Z21.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='b4'):
                ser.write(('G1 X-10.5 Y19 Z18 F1000\n').encode())
                ser.write(('G1 X-10.5 Y11.5 Z24.5 F1000\n').encode())
                ser.write(('G1 X-10.5 Y7.5 Z29.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-10.5 Y11.5 Z24.5 F1000\n').encode())
                ser.write(('G1 X-10.5 Y19 Z18 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='b5'):
                ser.write(('G1 X-11.5 Y17 Z14 F1000\n').encode())
                ser.write(('G1 X-11.5 Y8.5 Z20.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y3 Z27 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-11.5 Y8.5 Z20.5 F1000\n').encode())
                ser.write(('G1 X-11.5 Y17 Z14 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='b6'):
                ser.write(('G1 X-13 Y15 Z10 F1000\n').encode())
                ser.write(('G1 X-12.5 Y5 Z18 F1000\n').encode())
                ser.write(('G1 X-12.5 Y-1.5 Z25 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-12.5 Y5 Z18 F1000\n').encode())
                ser.write(('G1 X-13 Y15 Z10 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='b7'):
                ser.write(('G1 X-14.5 Y11.5 Z7 F1000\n').encode())
                ser.write(('G1 X-14 Y2 Z15.5 F1000\n').encode())
                ser.write(('G1 X-14 Y-5.5 Z23 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-14 Y2 Z15.5 F1000\n').encode())
                ser.write(('G1 X-14.5 Y11.5 Z7 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='b8'):
                ser.write(('G1 X-16.5 Y6.5 Z5.5 F1000\n').encode())
                ser.write(('G1 X-16.5 Y-1.5 Z13 F1000\n').encode())
                ser.write(('G1 X-16.5 Y-10 Z22.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-16.5 Y-1.5 Z13 F1000\n').encode())
                ser.write(('G1 X-16.5 Y6.5 Z5.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='c1'):
                ser.write(('G1 X-5.5 Y30 Z30 F1000\n').encode())
                ser.write(('G1 X-5 Y24 Z34.5 F1000\n').encode())
                ser.write(('G1 X-5.5 Y21 Z38 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-5 Y24 Z34.5 F1000\n').encode())
                ser.write(('G1 X-5.5 Y30 Z30 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='c2'):
                ser.write(('G1 X-6 Y25.5 Z25 F1000\n').encode())
                ser.write(('G1 X-6 Y19.5 Z30 F1000\n').encode())
                ser.write(('G1 X-6 Y15.5 Z34 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-6 Y19.5 Z30 F1000\n').encode())
                ser.write(('G1 X-6 Y25.5 Z25 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='c3'):
                ser.write(('G1 X-6 Y21 Z21 F1000\n').encode())
                ser.write(('G1 X-6 Y14.5 Z26.5 F1000\n').encode())
                ser.write(('G1 X-6 Y10.5 Z31 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-6 Y14.5 Z26.5 F1000\n').encode())
                ser.write(('G1 X-6 Y21 Z21 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='c4'):
                ser.write(('G1 X-6.5 Y18 Z17 F1000\n').encode())
                ser.write(('G1 X-6.5 Y10.5 Z23.5 F1000\n').encode())
                ser.write(('G1 X-6.5 Y6 Z29 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-6.5 Y10.5 Z23.5 F1000\n').encode())
                ser.write(('G1 X-6.5 Y18 Z17 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='c5'):
                ser.write(('G1 X-7 Y14.5 Z13.5 F1000\n').encode())
                ser.write(('G1 X-7 Y7 Z20 F1000\n').encode())
                ser.write(('G1 X-7 Y2 Z26.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-7 Y7 Z20 F1000\n').encode())
                ser.write(('G1 X-7 Y14.5 Z13.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='c6'):
                ser.write(('G1 X-8.5 Y14 Z8.5 F1000\n').encode())
                ser.write(('G1 X-8 Y3.5 Z17.5 F1000\n').encode())
                ser.write(('G1 X-8 Y-2.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-8 Y3.5 Z17.5 F1000\n').encode())
                ser.write(('G1 X-8.5 Y14 Z8.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='c7'):
                ser.write(('G1 X-9.5 Y8.5 Z6.5 F1000\n').encode())
                ser.write(('G1 X-9 Y-0.5 Z15 F1000\n').encode())
                ser.write(('G1 X-9 Y-7 Z22.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-9 Y-0.5 Z15 F1000\n').encode())
                ser.write(('G1 X-9.5 Y8.5 Z6.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='c8'):
                ser.write(('G1 X-10 Y5 Z4.5 F1000\n').encode())
                ser.write(('G1 X-10 Y-3.5 Z12 F1000\n').encode())
                ser.write(('G1 X-10.5 Y-12.5 Z22 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-10 Y-3.5 Z12 F1000\n').encode())
                ser.write(('G1 X-10 Y5 Z4.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='d1'):
                ser.write(('G1 X-3 Y28.5 Z30 F1000\n').encode())
                ser.write(('G1 X-2.5 Y23 Z34 F1000\n').encode())
                ser.write(('G1 X-2.5 Y20.5 Z37 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y23 Z34 F1000\n').encode())
                ser.write(('G1 X-3 Y28.5 Z30 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='d2'):
                ser.write(('G1 X-2.5 Y24.5 Z24.5 F1000\n').encode())
                ser.write(('G1 X-2.5 Y19 Z29 F1000\n').encode())
                ser.write(('G1 X-2.5 Y15 Z33.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y19 Z29 F1000\n').encode())
                ser.write(('G1 X-2.5 Y24.5 Z24.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='d3'):
                ser.write(('G1 X-2.5 Y21 Z20 F1000\n').encode())
                ser.write(('G1 X-2.5 Y14 Z26 F1000\n').encode())
                ser.write(('G1 X-2.5 Y10 Z30.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y14 Z26 F1000\n').encode())
                ser.write(('G1 X-2.5 Y21 Z20 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='d4'):
                ser.write(('G1 X-2.5 Y17.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X-2.5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X-3 Y6 Z28.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X-2.5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X-2.5 Y17.5 Z16.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='d5'):
                ser.write(('G1 X-3 Y14.5 Z13 F1000\n').encode())
                ser.write(('G1 X-2.5 Y7 Z19.5 F1000\n').encode())
                ser.write(('G1 X-3 Y1.5 Z26 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-2.5 Y7 Z19.5 F1000\n').encode())
                ser.write(('G1 X-3 Y14.5 Z13 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='d6'):
                ser.write(('G1 X-3.5 Y11.5 Z9.5 F1000\n').encode())
                ser.write(('G1 X-3 Y3 Z17 F1000\n').encode())
                ser.write(('G1 X-3 Y-3 Z24 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-3 Y3 Z17 F1000\n').encode())
                ser.write(('G1 X-3.5 Y11.5 Z9.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='d7'):
                ser.write(('G1 X-4 Y7 Z6.5 F1000\n').encode())
                ser.write(('G1 X-3.5 Y-0.5 Z13.5 F1000\n').encode())
                ser.write(('G1 X-3.5 Y-8 Z22 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-3.5 Y-0.5 Z13.5 F1000\n').encode())
                ser.write(('G1 X-4 Y7 Z6.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='d8'):
                ser.write(('G1 X-4.5 Y6 Z3 F1000\n').encode())
                ser.write(('G1 X-4 Y-7 Z15 F1000\n').encode())
                ser.write(('G1 X-4 Y-13.5 Z22 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X-4 Y-7 Z15 F1000\n').encode())
                ser.write(('G1 X-4.5 Y6 Z3 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='e1'):
                ser.write(('G1 X0.5 Y28.5 Z29.5 F1000\n').encode())
                ser.write(('G1 X0.5 Y23 Z35 F1000\n').encode())
                ser.write(('G1 X0.5 Y20.5 Z37 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X0.5 Y23 Z35 F1000\n').encode())
                ser.write(('G1 X0.5 Y28.5 Z29.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='e2'):
                ser.write(('G1 X0.5 Y24.5 Z24.5 F1000\n').encode())
                ser.write(('G1 X0.5 Y19 Z28.5 F1000\n').encode())
                ser.write(('G1 X0.5 Y15 Z33.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X0.5 Y19 Z28.5 F1000\n').encode())
                ser.write(('G1 X0.5 Y24.5 Z24.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='e3'):
                ser.write(('G1 X1 Y21 Z20 F1000\n').encode())
                ser.write(('G1 X1 Y14 Z25.5 F1000\n').encode())
                ser.write(('G1 X1 Y10 Z30 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X1 Y14 Z25.5 F1000\n').encode())
                ser.write(('G1 X1 Y21 Z20 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='e4'):
                ser.write(('G1 X1 Y17.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X1.5 Y10 Z23 F1000\n').encode())
                ser.write(('G1 X1.5 Y5.5 Z28 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X1.5 Y10 Z23 F1000\n').encode())
                ser.write(('G1 X1 Y17.5 Z16.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='e5'):
                ser.write(('G1 X1 Y15 Z12 F1000\n').encode())
                ser.write(('G1 X1.5 Y7 Z19 F1000\n').encode())
                ser.write(('G1 X1.5 Y1 Z26 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X1.5 Y7 Z19 F1000\n').encode())
                ser.write(('G1 X1 Y15 Z12 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='e6'):
                ser.write(('G1 X1.5 Y11 Z9.5 F1000\n').encode())
                ser.write(('G1 X2 Y3 Z16.5 F1000\n').encode())
                ser.write(('G1 X2 Y-3 Z24 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X2 Y3 Z16.5 F1000\n').encode())
                ser.write(('G1 X1.5 Y11 Z9.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='e7'):
                ser.write(('G1 X2 Y7 Z6.5 F1000\n').encode())
                ser.write(('G1 X2 Y0 Z13 F1000\n').encode())
                ser.write(('G1 X2 Y-8 Z22 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X2 Y0 Z13 F1000\n').encode())
                ser.write(('G1 X2 Y7 Z6.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='e8'):
                ser.write(('G1 X2.5 Y4.5 Z3.5 F1000\n').encode())
                ser.write(('G1 X3 Y-6 Z13 F1000\n').encode())
                ser.write(('G1 X3 Y-14 Z21.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X3 Y-6 Z13 F1000\n').encode())
                ser.write(('G1 X2.5 Y4.5 Z3.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='f1'):
                ser.write(('G1 X3.5 Y29 Z30 F1000\n').encode())
                ser.write(('G1 X3.5 Y23.5 Z34 F1000\n').encode())
                ser.write(('G1 X3.5 Y20.5 Z37.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X3.5 Y23.5 Z34 F1000\n').encode())
                ser.write(('G1 X3.5 Y29 Z30 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='f2'):
                ser.write(('G1 X4 Y25 Z25 F1000\n').encode())
                ser.write(('G1 X4 Y19 Z29.5 F1000\n').encode())
                ser.write(('G1 X4 Y15.5 Z33.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X4 Y19 Z29.5 F1000\n').encode())
                ser.write(('G1 X4 Y25 Z25 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='f3'):
                ser.write(('G1 X4.5 Y21.5 Z20 F1000\n').encode())
                ser.write(('G1 X4.5 Y14.5 Z25.5 F1000\n').encode())
                ser.write(('G1 X4.5 Y11 Z30 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X4.5 Y14.5 Z25.5 F1000\n').encode())
                ser.write(('G1 X4.5 Y21.5 Z20 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='f4'):
                ser.write(('G1 X5 Y19 Z16 F1000\n').encode())
                ser.write(('G1 X5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X5 Y6 Z28 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X5 Y10.5 Z23 F1000\n').encode())
                ser.write(('G1 X5 Y19 Z16 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='f5'):
                ser.write(('G1 X5.5 Y15.5 Z12.5 F1000\n').encode())
                ser.write(('G1 X6 Y7.5 Z19 F1000\n').encode())
                ser.write(('G1 X6 Y1.5 Z26 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X6 Y7.5 Z19 F1000\n').encode())
                ser.write(('G1 X5.5 Y15.5 Z12.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='f6'):
                ser.write(('G1 X6.5 Y11.5 Z9.5 F1000\n').encode())
                ser.write(('G1 X7 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X7 Y-2.5 Z24 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X7 Y3.5 Z16.5 F1000\n').encode())
                ser.write(('G1 X6.5 Y11.5 Z9.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='f7'):
                ser.write(('G1 X7.5 Y6 Z8 F1000\n').encode())
                ser.write(('G1 X8 Y-0.5 Z14 F1000\n').encode())
                ser.write(('G1 X8 Y-7.5 Z22 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X8 Y-0.5 Z14 F1000\n').encode())
                ser.write(('G1 X7.5 Y6 Z8 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='f8'):
                ser.write(('G1 X8.5 Y7.5 Z2 F1000\n').encode())
                ser.write(('G1 X9 Y-4 Z12 F1000\n').encode())
                ser.write(('G1 X9 Y-12.5 Z21 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X9 Y-4 Z12 F1000\n').encode())
                ser.write(('G1 X8.5 Y7.5 Z2 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='g1'):
                ser.write(('G1 X6 Y30 Z31.5 F1000\n').encode())
                ser.write(('G1 X6 Y25 Z35 F1000\n').encode())
                ser.write(('G1 X6 Y21.5 Z37.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X6 Y25 Z35 F1000\n').encode())
                ser.write(('G1 X6 Y30 Z31.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='g2'):
                ser.write(('G1 X7 Y25 Z25.5 F1000\n').encode())
                ser.write(('G1 X7 Y20 Z29.5 F1000\n').encode())
                ser.write(('G1 X7 Y16.5 Z34 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X7 Y20 Z29.5 F1000\n').encode())
                ser.write(('G1 X7 Y25 Z25.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='g3'):
                ser.write(('G1 X8 Y21.5 Z21 F1000\n').encode())
                ser.write(('G1 X8 Y15 Z26.5 F1000\n').encode())
                ser.write(('G1 X8 Y11.5 Z30.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X8 Y15 Z26.5 F1000\n').encode())
                ser.write(('G1 X8 Y21.5 Z21 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='g4'):
                ser.write(('G1 X9 Y18.5 Z17 F1000\n').encode())
                ser.write(('G1 X9 Y11.5 Z23 F1000\n').encode())
                ser.write(('G1 X9 Y6.5 Z28.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X9 Y11.5 Z23 F1000\n').encode())
                ser.write(('G1 X9 Y18.5 Z17 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='g5'):
                ser.write(('G1 X9.5 Y15.5 Z13.5 F1000\n').encode())
                ser.write(('G1 X10 Y8 Z20 F1000\n').encode())
                ser.write(('G1 X10 Y2.5 Z26 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X10 Y8 Z20 F1000\n').encode())
                ser.write(('G1 X9.5 Y15.5 Z13.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='g6'):
                ser.write(('G1 X11.5 Y12 Z10.5 F1000\n').encode())
                ser.write(('G1 X11.5 Y4 Z17.5 F1000\n').encode())
                ser.write(('G1 X11.5 Y-1.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X11.5 Y4 Z17.5 F1000\n').encode())
                ser.write(('G1 X11.5 Y12 Z10.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='g7'):
                ser.write(('G1 X13 Y7.5 Z8 F1000\n').encode())
                ser.write(('G1 X13 Y0.5 Z14.5 F1000\n').encode())
                ser.write(('G1 X13 Y-6.5 Z22.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X13 Y0.5 Z14.5 F1000\n').encode())
                ser.write(('G1 X13 Y7.5 Z8 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='g8'):
                ser.write(('G1 X16 Y6 Z4.5 F1000\n').encode())
                ser.write(('G1 X16 Y-2 Z12 F1000\n').encode())
                ser.write(('G1 X16 Y-8.5 Z19 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X16 Y-2 Z12 F1000\n').encode())
                ser.write(('G1 X16 Y6 Z4.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='h1'):
                ser.write(('G1 X9 Y30.5 Z32 F1000\n').encode())
                ser.write(('G1 X9 Y25.5 Z35.5 F1000\n').encode())
                ser.write(('G1 X9 Y23 Z39 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X9 Y25.5 Z35.5 F1000\n').encode())
                ser.write(('G1 X9 Y30.5 Z32 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='h2'):
                ser.write(('G1 X10 Y26 Z26.5 F1000\n').encode())
                ser.write(('G1 X10 Y21 Z30.5 F1000\n').encode())
                ser.write(('G1 X10 Y17.5 Z35 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X10 Y21 Z30.5 F1000\n').encode())
                ser.write(('G1 X10 Y26 Z26.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='h3'):
                ser.write(('G1 X11 Y21.5 Z22.5 F1000\n').encode())
                ser.write(('G1 X11 Y17 Z26 F1000\n').encode())
                ser.write(('G1 X11.5 Y13 Z31 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X11 Y17 Z26 F1000\n').encode())
                ser.write(('G1 X11 Y21.5 Z22.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='h4'):
                ser.write(('G1 X12.5 Y19.5 Z17.5 F1000\n').encode())
                ser.write(('G1 X12.5 Y13 Z23 F1000\n').encode())
                ser.write(('G1 X12.5 Y8.5 Z29 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                time.sleep(7)
                ser.write(('G1 X12.5 Y13 Z23 F1000\n').encode())
                ser.write(('G1 X12.5 Y19.5 Z17.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='h5'):
                ser.write(('G1 X13.5 Y17.5 Z14 F1000\n').encode())
                ser.write(('G1 X14 Y9 Z21 F1000\n').encode())
                ser.write(('G1 X14 Y4 Z27 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X14 Y9 Z21 F1000\n').encode())
                ser.write(('G1 X13.5 Y17.5 Z14 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='h6'):
                ser.write(('G1 X15.5 Y14 Z11 F1000\n').encode())
                ser.write(('G1 X16 Y6 Z18 F1000\n').encode())
                ser.write(('G1 X16 Y0.5 Z24.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X16 Y6 Z18 F1000\n').encode())
                ser.write(('G1 X15.5 Y14 Z11 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='h7'):
                ser.write(('G1 X17.5 Y9.5 Z8.5 F1000\n').encode())
                ser.write(('G1 X18 Y2 Z16 F1000\n').encode())
                ser.write(('G1 X18 Y-4 Z22.5 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X18 Y2 Z16 F1000\n').encode())
                ser.write(('G1 X17.5 Y9.5 Z8.5 F1000\n').encode())
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())
            elif(y=='h8'):
                ser.write(('G1 X21 Y9.5 Z3.5 F1000\n').encode())
                ser.write(('G1 X21 Y1.5 Z11.5 F1000\n').encode())
                ser.write(('G1 X21.5 Y-6.5 Z20 F1000\n').encode())
                ser.write(('M3 S45\n').encode()) #otworzenie
                ser.write(('G1 X21 Y1.5 Z11.5 F1000\n').encode())
                ser.write(('G1 X21 Y9.5 Z3.5 F1000\n').encode())
                time.sleep(10)
                time.sleep(5)
                ser.write(('G1 X0 Y0 Z0 F1000\n').encode())

            time.sleep(2)

    wyswietlanie()
    cv2.imshow("oryginal", frame)

    # Wykonaj 
    board.push(move)
    wyswietlanie()
    _, frame = capture.read()
    cv2.imshow("oryginal", frame)

# Wyświetl wynik końcowy
display_board(board)
print("Wynik:", board.result())

# Zamknij silnik Stockfish
engine.quit()

