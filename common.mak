CFLAGS += $(shell pkg-config --libs opencv)
CFLAGS += $(shell pkg-config --cflags opencv)

$(TARGET_FN) : $(SRC)
	$(CXX) $(SRC) -o $(SRC:.cpp=) $(CFLAGS)

clean:
	rm -rf $(TARGET_FN)
