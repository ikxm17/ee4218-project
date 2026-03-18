network:
  version: 2
  renderer: networkd
  ethernets:
    eth0:
      dhcp4: false
      addresses:
        - __IP_ADDRESS__/24
      routes:
        - to: default
          via: __GATEWAY__
      nameservers:
        addresses:
          - 8.8.8.8
          - 1.1.1.1
