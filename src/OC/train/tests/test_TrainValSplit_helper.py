SORTED_TEST_PAIRS = [
    (15, "sandwich", "sandwitch", 0.03), # 0
    (14, "meat", "meet", 0.24), # 2
    (13, "xxxy", "yyyy", 0.22), # 2
    (12, "stop", "pop", 0.3), # 3

    (10, "musical", "music", 0.16), # 1
    (10, "cccd", "cccc", 0.25), # 2

    (9, "lion", "leon", 0.25), # 2
    (9, "aabb", "aaaa", 0.4), # 3

    (8, "snake", "serpiente", 0.4), # 3
    (8, "giraffe", "jirafa", 0.03), # 0
    (8, "elephant", "elefante", 0.2), # 2

    (7, "orange", "naranja", 0.4), # 3
    (7, "happee", "happy", 0.2), # 2
    (7, "fridge", "freza", 0.16), # 1

    (6, "legion", "lechion", 0.05), # 0
    (6, "faint", "fanta", 0.05), # 0

    (5, "zebra", "zebra", 0.3), # 3
    (5, "ten", "dies", 0.4), # 3
    (5, "party", "fiesta", 0.4), # 3
    (5, "hello", "heko", 0.2), # 2
    
    (4, "miracle", "milagro", 0.33), # 3
    (4, "mermaid", "merman", 0.4), # 3
    (4, "mercantile", "domicile", 0.12), # 1
    (4, "horse", "herse", 0.04), # 0
    (4, "animals", "animales", 0.13), # 1

    (3, "stomach", "estomago", 0.28), # 2
    (3, "pizza", "pisa?", 0.17), # 1
    (3, "mucho", "much", 0.2), # 2

    (2, "ocean", "oceano", 0.17), # 1
    (2, "cookies", "biscuits", 0.4), # 3

    (1, "smoke", "stove", 0.3), # 3
    (1, "hungry", "hippo", 0.07), # 0
]




TEST_PAIRS = [
    (5, "hello", "heko", 0.2), # 2
    (3, "pizza", "pisa?", 0.17), # 1
    (5, "zebra", "zebra", 0.3), # 3
    (10, "musical", "music", 0.16), # 1
    (1, "hungry", "hippo", 0.07), # 0
    (6, "legion", "lechion", 0.05), # 0
    (7, "happee", "happy", 0.2), # 2
    (9, "aabb", "aaaa", 0.4), # 3
    (10, "cccd", "cccc", 0.25), # 2
    (13, "xxxy", "yyyy", 0.22), # 2
    (7, "orange", "naranja", 0.4), # 3
    (4, "miracle", "milagro", 0.33), # 3
    (2, "ocean", "oceano", 0.17), # 1
    (8, "elephant", "elefante", 0.2), # 2
    (9, "lion", "leon", 0.25), # 2
    (4, "mercantile", "domicile", 0.12), # 1
    (4, "animals", "animales", 0.13), # 1
    (3, "mucho", "much", 0.2), # 2
    (4, "horse", "herse", 0.04), # 0
    (5, "party", "fiesta", 0.4), # 3
    (7, "fridge", "freza", 0.16), # 1
    (8, "giraffe", "jirafa", 0.03), # 0
    (1, "smoke", "stove", 0.3), # 3
    (2, "cookies", "biscuits", 0.4), # 3
    (4, "mermaid", "merman", 0.4), # 3
    (6, "faint", "fanta", 0.05), # 0
    (8, "snake", "serpiente", 0.4), # 3
    (5, "ten", "dies", 0.4), # 3
    (3, "stomach", "estomago", 0.28), # 2
    (15, "sandwich", "sandwitch", 0.03), # 0
    (14, "meat", "meet", 0.24), # 2
    (12, "stop", "pop", 0.3), # 3
]

# if SORTED_TEST_PAIRS == sorted(TEST_PAIRS, reverse=True):
#     print("They are the same")
# else:
#     print("different!!!!")





TEST_PAIRS_CAP = [
    (15, 'sandwich', 'sandwitch', 0.03), # 0
    (14, 'meat', 'meet', 0.24), # 2
    (13, 'xxxy', 'yyyy', 0.22), # 2
    (12, 'stop', 'pop', 0.31), # 3

    (10, 'musical', 'music', 0.16), # 1
    (10, 'cccd', 'cccc', 0.25), # 2

    (9, 'lion', 'leon', 0.25), # 2
    (9, 'aabb', 'aaaa', 0.4), # 3

    (8, 'snake', 'serpiente', 0.4), # 3
    (8, 'giraffe', 'jirafa', 0.26), # 2
    (8, 'elephant', 'elefante', 0.21), # 2

    (7, 'orange', 'naranja', 0.4), # 3
    (7, 'happee', 'happy', 0.21), # 2
    (7, 'fridge', 'freza', 0.16), # 1

    (6, 'legion', 'lechion', 0.25), # 2
    (6, 'faint', 'fanta', 0.26), # 2

    (5, 'zebra', 'zebra', 0.31), # 3
    (5, 'ten', 'dies', 0.4), # 3
    (5, 'party', 'fiesta', 0.4), # 3
    (5, 'hello', 'heko', 0.21), # 2

    (4, 'miracle', 'milagro', 0.23), # 2
    (4, 'mermaid', 'merman', 0.4), # 3
    (4, 'mercantile', 'domicile', 0.12), # 1
    (4, 'horse', 'herse', 0.04), # 0
    (4, 'animals', 'animales', 0.23), # 2

    (3, 'stomach', 'estomago', 0.28), # 2
    (3, 'pizza', 'pisa?', 0.17), # 1
    (3, 'mucho', 'much', 0.21), # 2

    (2, 'ocean', 'oceano', 0.17), # 1
    (2, 'cookies', 'biscuits', 0.4), # 3

    (1, 'smoke', 'stove', 0.21), # 2
    (1, 'hungry', 'hippo', 0.07) # 0
]




TEST_PAIRS_CAP_DEFECIT = [
    (15, 'sandwich', 'sandwitch', 0.03), # 0
    (14, 'meat', 'meet', 0.24), # 2
    (13, 'xxxy', 'yyyy', 0.22), # 2

    (10, 'musical', 'music', 0.16), # 1
    (10, 'cccd', 'cccc', 0.25), # 2

    (9, 'lion', 'leon', 0.25), # 2
    (9, 'aabb', 'aaaa', 0.4), # 3

    (8, 'snake', 'serpiente', 0.4), # 3

    (7, 'orange', 'naranja', 0.4), # 3
    (7, 'fridge', 'freza', 0.16), # 1

    (5, 'zebra', 'zebra', 0.31), # 3
    (5, 'ten', 'dies', 0.4), # 3
    (5, 'hello', 'heko', 0.21), # 2

    (4, 'miracle', 'milagro', 0.23), # 2
    (4, 'mercantile', 'domicile', 0.12), # 1
    (4, 'horse', 'herse', 0.04), # 0
    (4, 'animals', 'animales', 0.23), # 2

    (3, 'pizza', 'pisa?', 0.17), # 1
    (3, 'mucho', 'much', 0.21), # 2

    (2, 'ocean', 'oceano', 0.17), # 1
    (2, 'cookies', 'biscuits', 0.4), # 3

    (1, 'smoke', 'stove', 0.21), # 2
    (1, 'hungry', 'hippo', 0.07) # 0
]



# This should have 6 pairs removed becuase of duplicates
TEST_PAIRS_FUZZ = [
    (14, 15, 'meat', 'meet', 0.24), # 2
    (13, 10, 'xxxy', 'yyyy', 0.22), # 2
    (12, 10, 'stop', 'pop', 0.31), # 3
    (15, 8, 'sandwich', 'sandwitch', 0.03), # 0
    (10, 9, 'musical', 'music', 0.16), # 1
    (9, 8, 'lion', 'leon', 0.25), # 2
    (10, 7, 'cccd', 'cccc', 0.25), # 2
    (8, 7, 'snake', 'serpiente', 0.4), # 3
    (8, 6, 'elephant', 'elefante', 0.21), # 2
    (9, 5, 'aabb', 'aaaa', 0.4), # 3
    (7, 6, 'orange', 'naranja', 0.4), # 3
    (6, 7, 'legion', 'lechion', 0.05), # 0
    (5, 8, 'party', 'fiesta', 0.4), # 3
    (6, 6, 'faint', 'fanta', 0.05), # 0
    (5, 6, 'zebra', 'zebra', 0.31), # 3
    (7, 4, 'happee', 'happy', 0.21), # 2
    (4, 6, 'mermaid', 'merman', 0.4), # 3
    (5, 4, 'ten', 'dies', 0.4), # 3
    (4, 5, 'mercantile', 'domicile', 0.12), # 1
    (4, 5, 'animals', 'animales', 0.13), # 1
    (3, 6, 'mucho', 'much', 0.21), # 2
    (2, 8, 'cookies', 'biscuits', 0.4), # 3
    (7, 2, 'fridge', 'freza', 0.16), # 1
    (4, 3, 'miracle', 'milagro', 0.33), # 3
    (4, 3, 'horse', 'herse', 0.04), # 0
    (3, 4, 'pizza', 'pisa?', 0.17), # 1
    (5, 2, 'hello', 'heko', 0.21), # 2
    (8, 1, 'giraffe', 'jirafa', 0.03), # 0
    (2, 4, 'ocean', 'oceano', 0.17), # 1
    (3, 2, 'stomach', 'estomago', 0.28), # 2
    (1, 5, 'smoke', 'stove', 0.31), # 3
    (1, 4, 'hungry', 'hippo', 0.07) # 0
]