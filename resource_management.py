class Resource:
    def __init__(self, name, total_available, renewable):
        self.name = name
        self.total_available = total_available
        self.renewable = renewable

    def report_usage(self):
        print(f"Resource: {self.name}")
        print(f"Total Available: {self.total_available}")
        print(f"Renewable: {'Yes' if self.renewable else 'No'}")

    def update_availability(self, amount):
        if amount > self.total_available:
            raise ValueError("Insufficient resource available.")
        self.total_available -= amount

class WaterResource(Resource):
    def __init__(self, total_available):
        super().__init__("Water", total_available, renewable=True)

class EnergyResource(Resource):
    def __init__(self, total_available, renewable):
        super().__init__("Energy", total_available, renewable)

class WasteResource(Resource):
    def __init__(self, total_available):
        super().__init__("Waste", total_available, renewable=False)

class Consumer:
    def __init__(self, consumer_id, name):
        self.consumer_id = consumer_id
        self.name = name
        self.assigned_resources = []

    def assign_resource(self, resource):
        self.assigned_resources.append(resource)

    def use_resource(self, resource_name, amount):
        for resource in self.assigned_resources:
            if resource.name == resource_name:
                resource.update_availability(amount)
                print(f"{amount} of {resource_name} used by {self.name}.")
                return
        raise ValueError(f"Resource {resource_name} not assigned to {self.name}.")

    def generate_usage_report(self):
        print(f"Usage Report for {self.name}:")
        for resource in self.assigned_resources:
            resource.report_usage()

# Sample interaction
def main():
    # Create resources
    water = WaterResource(1000)
    solar_energy = EnergyResource(500, renewable=True)
    coal_energy = EnergyResource(300, renewable=False)
    waste = WasteResource(200)

    # Create consumers
    household = Consumer(1, "Household")
    factory = Consumer(2, "Factory")

    # Assign resources to consumers
    household.assign_resource(water)
    household.assign_resource(solar_energy)
    factory.assign_resource(coal_energy)
    factory.assign_resource(waste)

    # Use resources
    household.use_resource("Water", 100)
    household.use_resource("Energy", 50)
    factory.use_resource("Energy", 100)
    factory.use_resource("Waste", 50)

    # Generate reports
    household.generate_usage_report()
    factory.generate_usage_report()

if __name__ == "__main__":
    main()