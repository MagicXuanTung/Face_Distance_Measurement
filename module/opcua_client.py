from opcua import Client, ua


class OPCUAClient:
    def __init__(self, url):
        self.url = url
        self.client = Client(url)

    def connect(self):
        try:
            self.client.connect()
            print("Connected to Kepware OPC UA Server")
        except Exception as e:
            print(f"Failed to connect to OPC UA Server: {e}")
            exit()

    def disconnect(self):
        self.client.disconnect()

    def get_node(self, node_id):
        return self.client.get_node(node_id)

    def set_value(self, node, value, variant_type=ua.VariantType.UInt16):
        try:
            node.set_value(ua.DataValue(ua.Variant(value, variant_type)))
        except Exception as e:
            print(f"Failed to set value for node {node}: {e}")
