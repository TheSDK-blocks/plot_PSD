add wave -position insertpoint  \
sim/:tb_myentity:A \
sim/:tb_myentity:initdone \
sim/:tb_myentity:clock \
sim/:tb_myentity:Z \

run -all
