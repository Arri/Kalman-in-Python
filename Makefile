PROJECT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

SRCDIR := $(PROJECT_DIR)/src
OBJDIR := obj

SRCS := $(shell find $(SRCDIR) -type f -name "*.c") 
OBJS := $(patsubst $(SRCDIR)/%,$(OBJDIR)/%.o,$(SRCS))

TARGET := kalman

INCDIRS += $(SRCDIR)

CFLAGS += -Wall -Wextra
CFLAGS += -g -O0
CFLAGS += $(addprefix -I,$(INCDIRS))

all: $(TARGET)
.PHONY: all

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) -o $@ $^ $(LDFLAGS)

$(OBJS): $(OBJDIR)/%.o: $(SRCDIR)/%
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -f $(OBJS)
.PHONY: clean

fclean: clean
	rm -f $(TARGET)
.PHONY: fclean

re: fclean all
.PHONY: re
